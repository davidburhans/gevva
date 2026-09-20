"""Gemma 4 Multimodal NLI Cross-Encoder / Decision Model.

This module adapts the Google Gemma 4 multimodal architecture (`google/gemma-4-E2B`
and `google/gemma-4-E4B`) into a high-throughput, large-context, vision-enabled
NLI Cross-Encoder.

Label Ordering follows dleemiller / ModernCE / OpenJEV standard:
    0: contradiction
    1: entailment
    2: neutral

References:
    - OpenJEV: research/openjev/modeling_openjev.py
    - Laya: research/laya/rl_agent_api.py
    - Gemma 4 Configs: google/gemma-4-E2B, google/gemma-4-E4B
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Gemma4Config,
    Gemma4Model,
    Gemma4PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

# -----------------------------------------------------------------------------
# Constants & Label Mapping
# -----------------------------------------------------------------------------
CONTRADICTION: int = 0
ENTAILMENT: int = 1
NEUTRAL: int = 2

ID2LABEL: Dict[int, str] = {
    0: "contradiction",
    1: "entailment",
    2: "neutral",
}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}

# Standard SNLI / MNLI native format: 0=entailment, 1=neutral, 2=contradiction
NATIVE_MNLI2OURS: Dict[int, int] = {0: 1, 1: 2, 2: 0}

DEFAULT_NLI_TEMPLATE: str = "Premise: {premise}\nHypothesis: {hypothesis}"


# -----------------------------------------------------------------------------
# Model Class: Gemma4ForSequenceClassification
# -----------------------------------------------------------------------------
class Gemma4ForSequenceClassification(Gemma4PreTrainedModel):
    """Gemma 4 backbone with a pooled Sequence Classification head for NLI and decision-making."""

    config_class = Gemma4Config
    base_model_prefix = "model"

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 3)
        self.model = Gemma4Model(config)

        # We normalize the pooled representation using Gemma 4's RMSNorm
        # to ensure training stability across heterogeneous layer configurations
        text_hidden_size = config.text_config.hidden_size
        self.norm = Gemma4RMSNorm(text_hidden_size, eps=config.text_config.rms_norm_eps)
        self.score = nn.Linear(text_hidden_size, self.num_labels, bias=False)

        # Initialize weights and post-init processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def freeze_vision_tower(self, freeze_adapter: bool = False):
        """Freezes vision encoder parameters to reduce memory and speed up training.

        Args:
            freeze_adapter: If False (recommended), the multimodal projection adapter
                            `model.embed_vision` remains trainable while the deep vision
                            encoder layers are frozen. If True, freezes the projection as well.
        """
        if self.model.vision_tower is not None:
            for p in self.model.vision_tower.parameters():
                p.requires_grad = False
            self.model.vision_tower.eval()

        if freeze_adapter and self.model.embed_vision is not None:
            for p in self.model.embed_vision.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        r"""
        Args:
            input_ids: Indices of input sequence tokens in vocabulary.
            pixel_values: Input image patch features `(num_images, max_patches, 3 * patch_size^2)`.
            image_position_ids: 2D coordinates for image patches `(num_images, max_patches, 2)`.
            attention_mask: Mask to avoid attention over padding tokens (1 = valid, 0 = pad).
            position_ids: Position indices.
            labels: Labels for computing classification cross-entropy loss `(batch_size,)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Gemma4Model handles multimodal token merging, PLE lookup, and hybrid attention masks
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, text_hidden_size)

        # Last non-pad token pooling:
        # In causal decoders, information flows left-to-right from premise through hypothesis.
        # Flip-argmax ensures exact invariance to left-padding, right-padding, or mixed orientations.
        if attention_mask is not None:
            reversed_mask = attention_mask.flip(dims=[-1])
            last_token_indices = (attention_mask.shape[-1] - 1 - reversed_mask.argmax(dim=-1)).long()
            batch_size = hidden_states.shape[0]
            pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_token_indices]
        else:
            pooled = hidden_states[:, -1]

        # Normalized classification projection
        pooled_normed = self.norm(pooled)
        logits = self.score(pooled_normed)  # (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward_packed(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        r"""Forward pass for packed variable-length sequences without padding overhead.
        
        Args:
            input_ids: 1D tensor `(total_tokens,)` or 2D `(1, total_tokens)` containing concatenated sequences.
            cu_seqlens: Cumulative sequence length indices `[0, L1, L1+L2, ..., total_tokens]`.
            labels: 1D tensor `(num_sequences,)` containing ground-truth targets.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # (1, total_tokens, text_hidden_size)

        # Multi-sequence terminal token gathering:
        # The terminal token of sequence k is located exactly at cu_seqlens[k+1] - 1
        terminal_indices = cu_seqlens[1:] - 1
        pooled = hidden_states[0, terminal_indices]  # (num_sequences, text_hidden_size)

        pooled_normed = self.norm(pooled)
        logits = self.score(pooled_normed)  # (num_sequences, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


# Register model in Hugging Face AutoModelForSequenceClassification
AutoModelForSequenceClassification.register(Gemma4Config, Gemma4ForSequenceClassification, exist_ok=True)


def tokenize_nli_pair_safe(
    tokenizer,
    premise: str,
    hypothesis: str,
    max_length: int = 2048,
    image_soft_tokens: int = 0,
    boi_token: str = "<|image>",
    image_token: str = "<|image|>",
    eoi_token: str = "<image|>",
) -> List[int]:
    r"""Safely tokenizes premise-hypothesis pair with budget-aware truncation.

    Guarantees:
    1. Hypothesis is strictly preserved (never truncated).
    2. Premise text is truncated from the trailing end only if total length exceeds max_length.
    3. Terminal prompt ends with fixed '\nPrediction:' token to eliminate token identity prior bias.
    4. Leading BOS token serves as dedicated attention sink for causal stability.
    """
    bos_id = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    hyp_ids = tokenizer.encode(f"\nHypothesis: {hypothesis.strip()}\nPrediction:", add_special_tokens=False)
    prem_prefix_ids = tokenizer.encode("Premise: ", add_special_tokens=False)

    vis_ids = []
    if image_soft_tokens > 0:
        boi_id = tokenizer.convert_tokens_to_ids(boi_token)
        img_id = tokenizer.convert_tokens_to_ids(image_token)
        eoi_id = tokenizer.convert_tokens_to_ids(eoi_token)
        if boi_id is not None and img_id is not None and eoi_id is not None:
            vis_ids = [boi_id] + [img_id] * image_soft_tokens + [eoi_id] + tokenizer.encode(" ", add_special_tokens=False)
        else:
            vis_ids = tokenizer.encode(f"{boi_token}{image_token * image_soft_tokens}{eoi_token} ", add_special_tokens=False)

    overhead = len(bos_id) + len(prem_prefix_ids) + len(vis_ids) + len(hyp_ids)
    avail_premise = max_length - overhead
    if avail_premise < 16:
        avail_premise = max(8, max_length - len(hyp_ids))

    prem_ids = tokenizer.encode(premise.strip(), add_special_tokens=False)[:max(0, avail_premise)]
    full_ids = bos_id + prem_prefix_ids + vis_ids + prem_ids + hyp_ids
    return full_ids


class RerankResult(int):
    """Subclass of int that preserves Jev integer index behavior while supporting score unpacking."""
    scores: List[float]

    def __new__(cls, best_idx: int, scores: List[float]):
        val = super().__new__(cls, best_idx)
        val.scores = scores
        return val

    def __iter__(self):
        yield int(self)
        yield self.scores

    def __getitem__(self, idx):
        return (int(self), self.scores)[idx]

    @property
    def index(self) -> int:
        return int(self)


class GradeResult(str):
    """Subclass of str that preserves Jev string verdict behavior while supporting probability unpacking."""
    probabilities: Dict[str, float]

    def __new__(cls, label: str, probabilities: Dict[str, float]):
        val = super().__new__(cls, label)
        val.probabilities = probabilities
        return val

    def __iter__(self):
        yield str(self)
        yield self.probabilities

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.probabilities[idx]
        return (str(self), self.probabilities)[idx]

    @property
    def label(self) -> str:
        return str(self)


# -----------------------------------------------------------------------------
# High-Level Cross-Encoder Interface
# -----------------------------------------------------------------------------
class Gemma4CrossEncoder:
    """High-level wrapper for inference, reranking, grading, and latent extraction.
    
    Fully compatible with the TypeSafe AI / openjev cross-encoder API and options.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
        batch_size: int = 16,
        template: str = DEFAULT_NLI_TEMPLATE,
        # Jev parameter aliases:
        path: Optional[str] = None,
        subfolder: Optional[str] = None,
        bs: Optional[int] = None,
        max_len: Optional[int] = None,
        **kwargs,
    ):
        model_name_or_path = model_name_or_path or path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_length = max_len if max_len is not None else max_length
        self.batch_size = bs if bs is not None else batch_size
        self.template = template
        self.bs = self.batch_size
        self.max_len = self.max_length

        # Tokenizer initialization (Right-padding is mandatory for last-token pooling)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name_or_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            raise ValueError("Must provide either model_name_or_path or tokenizer.")

        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.pad_token_id = 0
        self.tok = self.tokenizer

        # Image Processor
        self.image_processor = Gemma4ImageProcessorPil()

        # Model initialization
        if model is not None:
            self.model = model
        elif model_name_or_path is not None:
            quant_cfg_file = os.path.join(model_name_or_path, "quantization_config.json")
            weights_file = os.path.join(model_name_or_path, "model.safetensors")
            adapter_cfg_file = os.path.join(model_name_or_path, "adapter_config.json")

            if os.path.exists(quant_cfg_file) and os.path.exists(weights_file):
                print(f"Loading W4A16 packed model from {model_name_or_path}...")
                from safetensors.torch import load_file
                from export_w4a16 import unpack_weights_w4a16
                with open(quant_cfg_file, "r") as f:
                    qcfg = json.load(f)
                group_size = qcfg.get("config_groups", {}).get("group_0", {}).get("weights", {}).get("group_size", 32)

                base_cfg = AutoConfig.from_pretrained(model_name_or_path)
                base_cfg.num_labels = 3
                with torch.device("meta"):
                    base_model = Gemma4ForSequenceClassification(base_cfg).to(dtype=dtype)
                base_model = base_model.to_empty(device=self.device)

                packed_sd = load_file(weights_file, device="cpu")
                model_tensors = {**dict(base_model.named_parameters()), **dict(base_model.named_buffers())}
                for k, tensor in packed_sd.items():
                    if k.endswith(".weight_packed"):
                        base_key = k[: -len(".weight_packed")]
                        target_key = f"{base_key}.weight"
                        if target_key in model_tensors:
                            scale = packed_sd[f"{base_key}.weight_scale"].to(self.device)
                            packed = tensor.to(self.device)
                            unpacked = unpack_weights_w4a16(
                                packed, scale, group_size=group_size, dtype=dtype
                            )
                            model_tensors[target_key].data.copy_(unpacked)
                            del packed, scale, unpacked
                    elif k.endswith(".weight_scale") or k.endswith(".weight_shape"):
                        continue
                    else:
                        if k in model_tensors:
                            model_tensors[k].data.copy_(tensor.to(dtype=dtype))
                del packed_sd, model_tensors

                # Restore non-persistent buffers (RoPE frequencies and embedding scales) uninitialized by meta-device
                lm = getattr(base_model.model, "language_model", getattr(base_model, "language_model", getattr(base_model, "model", base_model)))
                if hasattr(lm, "rotary_emb"):
                    lm.rotary_emb.__init__(lm.rotary_emb.config)
                    lm.rotary_emb.to(self.device)
                for mod in base_model.modules():
                    if hasattr(mod, "scalar_embed_scale") and hasattr(mod, "embed_scale"):
                        mod.embed_scale.data.copy_(torch.tensor(mod.scalar_embed_scale, dtype=dtype, device=self.device))

                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()
                self.model = base_model
            elif os.path.exists(adapter_cfg_file):
                with open(adapter_cfg_file, "r") as f:
                    acfg = json.load(f)
                base_id = acfg.get("base_model_name_or_path", "google/gemma-4-E2B")
                base_config = AutoConfig.from_pretrained(base_id)
                base_config.num_labels = 3
                if "gemma" in base_id.lower():
                    base_model = Gemma4ForSequenceClassification.from_pretrained(
                        base_id,
                        config=base_config,
                        torch_dtype=dtype,
                    )
                else:
                    from transformers import AutoModelForSequenceClassification
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_id,
                        config=base_config,
                        torch_dtype=dtype,
                    )
                from peft import PeftModel
                peft_model = PeftModel.from_pretrained(base_model, model_name_or_path)
                head_weights_path = os.path.join(model_name_or_path, "head_weights.pt")
                if os.path.exists(head_weights_path):
                    hw = torch.load(head_weights_path, map_location="cpu", weights_only=True)
                    raw = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model

                    def _load_into_module(mod, state):
                        if not isinstance(state, dict):
                            w = state
                        elif "weight" in state:
                            w = state["weight"]
                        elif "modules_to_save.default.weight" in state:
                            w = state["modules_to_save.default.weight"]
                        elif "default.weight" in state:
                            w = state["default.weight"]
                        elif "original_module.weight" in state:
                            w = state["original_module.weight"]
                        else:
                            w = next(iter(state.values()))
                        if hasattr(mod, "modules_to_save") and "default" in mod.modules_to_save:
                            mod.modules_to_save["default"].weight.data.copy_(w)
                            if hasattr(mod, "original_module") and hasattr(mod.original_module, "weight"):
                                mod.original_module.weight.data.copy_(w)
                        elif hasattr(mod, "weight"):
                            mod.weight.data.copy_(w)
                        else:
                            mod.load_state_dict({"weight": w}, strict=False)

                    if "score" in hw and hasattr(raw, "score"):
                        _load_into_module(raw.score, hw["score"])
                    if "norm" in hw and hasattr(raw, "norm"):
                        _load_into_module(raw.norm, hw["norm"])
                self.model = peft_model
            else:
                if "gemma" in model_name_or_path.lower():
                    self.model = Gemma4ForSequenceClassification.from_pretrained(
                        model_name_or_path,
                        num_labels=3,
                        id2label=ID2LABEL,
                        label2id=LABEL2ID,
                        torch_dtype=dtype,
                    )
                else:
                    from transformers import AutoModelForSequenceClassification
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name_or_path,
                        num_labels=3,
                        id2label=ID2LABEL,
                        label2id=LABEL2ID,
                        torch_dtype=dtype,
                    )
        else:
            raise ValueError("Must provide either model_name_or_path or model.")

        self.model.to(self.device).eval()

        # Special token IDs for multimodal inputs
        self.boi_token_id = getattr(self.model.config, "boi_token_id", 255999)
        self.image_token_id = getattr(self.model.config, "image_token_id", 258880)
        self.eoi_token_id = getattr(self.model.config, "eoi_token_id", 258882)
        self.boi_token = "<|image>"
        self.image_token = "<|image|>"
        self.eoi_token = "<image|>"

    def _prepare_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Formats and tokenizes pairs, processing images where provided."""
        if images is None:
            images = [None] * len(pairs)

        batch_input_ids = []
        processed_pixel_values = []
        processed_position_ids = []

        for (premise, hypothesis), img in zip(pairs, images):
            n_soft = 0
            if img is not None:
                feat = self.image_processor(img, return_tensors="pt")
                n_soft = int(feat["num_soft_tokens_per_image"][0])
                processed_pixel_values.append(feat["pixel_values"][0])
                processed_position_ids.append(feat["image_position_ids"][0])

            ids = tokenize_nli_pair_safe(
                tokenizer=self.tokenizer,
                premise=premise,
                hypothesis=hypothesis,
                max_length=self.max_length,
                image_soft_tokens=n_soft,
                boi_token=self.boi_token,
                image_token=self.image_token,
                eoi_token=self.eoi_token,
            )
            batch_input_ids.append(ids)

        # Pad to max length in batch with right-padding
        max_batch_len = max(len(ids) for ids in batch_input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_ids = []
        attn_masks = []
        for ids in batch_input_ids:
            pad_len = max_batch_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        batch: Dict[str, Any] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long, device=self.device),
        }

        if processed_pixel_values:
            batch["pixel_values"] = torch.stack(processed_pixel_values).to(self.device, dtype=self.dtype)
            batch["image_position_ids"] = torch.stack(processed_position_ids).to(self.device)

        return batch

    @torch.no_grad()
    def predict(
        self,
        pairs: Sequence[Tuple[str, str]],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> np.ndarray:
        """Computes softmax probabilities [p_contradiction, p_entailment, p_neutral] per pair."""
        all_probs = []
        n_items = len(pairs)

        for s in range(0, n_items, self.batch_size):
            chunk_pairs = pairs[s : s + self.batch_size]
            chunk_images = images[s : s + self.batch_size] if images is not None else None
            batch = self._prepare_batch(chunk_pairs, chunk_images)

            logits = self.model(**batch).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 3))

    @torch.no_grad()
    def predict_logits(
        self,
        pairs: Sequence[Tuple[str, str]],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> np.ndarray:
        """Computes raw classification logits [z_contradiction, z_entailment, z_neutral] per pair."""
        all_logits = []
        n_items = len(pairs)

        for s in range(0, n_items, self.batch_size):
            chunk_pairs = pairs[s : s + self.batch_size]
            chunk_images = images[s : s + self.batch_size] if images is not None else None
            batch = self._prepare_batch(chunk_pairs, chunk_images)

            logits = self.model(**batch).logits.float().cpu().numpy()
            all_logits.append(logits)

        return np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 3))

    @torch.no_grad()
    def extract_latents(
        self,
        pairs: Sequence[Tuple[str, str]],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> np.ndarray:
        """Extracts the pooled representation vector for probing or auxiliary head training."""
        all_latents = []
        n_items = len(pairs)

        for s in range(0, n_items, self.batch_size):
            chunk_pairs = pairs[s : s + self.batch_size]
            chunk_images = images[s : s + self.batch_size] if images is not None else None
            batch = self._prepare_batch(chunk_pairs, chunk_images)

            outputs = self.model.model(
                input_ids=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                image_position_ids=batch.get("image_position_ids"),
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            # Flip-argmax pooling strictly invariant to padding side
            reversed_mask = batch["attention_mask"].flip(dims=[-1])
            last_indices = (batch["attention_mask"].shape[-1] - 1 - reversed_mask.argmax(dim=-1)).long()
            batch_size = hidden.shape[0]
            pooled = hidden[torch.arange(batch_size, device=hidden.device), last_indices]
            # Normalize using Gemma4RMSNorm matching the classification head
            norm_layer = getattr(self.model, "norm", None)
            if norm_layer is None and hasattr(self.model, "base_model"):
                norm_layer = getattr(self.model.base_model.model, "norm", None)
            if norm_layer is not None:
                pooled = norm_layer(pooled)
            all_latents.append(pooled.float().cpu().numpy())

        return np.concatenate(all_latents, axis=0) if all_latents else np.empty((0, self.model.config.text_config.hidden_size))

    # Jev API alias
    latents = extract_latents

    def rerank(
        self,
        premise: Optional[str] = None,
        options: Sequence[str] = (),
        image: Optional[Image.Image] = None,
        hyp_format: Optional[str] = None,
        # Jev parameter aliases:
        question: Optional[str] = None,
        hyp_fmt: Optional[str] = None,
        scoring: str = "margin",
        temperature: Optional[Union[float, str]] = None,
    ) -> RerankResult:
        """Reranks options by argmax score.
        
        if not options:
            raise ValueError("rerank requires at least one option (audit LOW-MED: empty-array crash)")
        
        Args:
            premise: Context or question string
            options: List of candidate answers or continuations
            image: Optional PIL Image for multimodal reranking
            hyp_format: Template for candidate hypotheses, e.g. "The correct answer is: {}"
            question: Alias for premise (Jev compatibility)
            hyp_fmt: Alias for hyp_format (Jev compatibility)
            scoring: Scoring rule:
                - 'margin' (recommended): argmax (P(entailment) - P(contradiction)), robust to neutral priors
                - 'log_odds' / 'logit_margin': argmax (z_ent - z_con), strictly invariant to neutral logit
                - 'entailment' (Jev legacy): argmax P(entailment)
                - 'contrastive': argmax (P(ent) / (P(ent) + P(con) + 1e-6))
            temperature: Optional temperature scaling (scalar float, or 'bucketed' for cardinality scaling)
        
        Fully compatible with Jev: returns RerankResult which behaves as an int index,
        while supporting tuple unpacking (best_idx, scores) and .scores attribute.
        """
        query = premise if premise is not None else question
        if query is None:
            raise ValueError("Must provide either premise or question.")
        fmt = hyp_format or hyp_fmt or "The correct answer is: {}"
        pairs = [(query, fmt.format(opt)) for opt in options]
        images = [image] * len(options) if image is not None else None

        if scoring in ("log_odds", "logit_margin"):
            logits = self.predict_logits(pairs, images)
            raw_scores = logits[:, ENTAILMENT] - logits[:, CONTRADICTION]
        elif scoring == "margin":
            probs = self.predict(pairs, images)
            raw_scores = probs[:, ENTAILMENT] - probs[:, CONTRADICTION]
        elif scoring == "contrastive":
            probs = self.predict(pairs, images)
            raw_scores = probs[:, ENTAILMENT] / (probs[:, ENTAILMENT] + probs[:, CONTRADICTION] + 1e-6)
        else:
            probs = self.predict(pairs, images)
            raw_scores = probs[:, ENTAILMENT]

        if temperature is not None:
            K = len(options)
            if temperature == "bucketed":
                if K == 2:
                    T = 1.637
                elif 3 <= K <= 5:
                    T = 1.251
                elif 6 <= K <= 10:
                    T = 1.420
                else:
                    T = 1.983
            else:
                T = float(temperature)
            scaled = raw_scores / max(T, 1e-4)
            # Softmax normalized option probabilities
            exp_s = np.exp(scaled - np.max(scaled))
            scores = (exp_s / np.sum(exp_s)).tolist()
        else:
            scores = raw_scores.tolist()

        best_idx = int(np.argmax(scores))
        return RerankResult(best_idx, scores)

    def grade(
        self,
        premise: Optional[str] = None,
        reference_answer: Optional[str] = None,
        candidate_answer: Optional[str] = None,
        image: Optional[Image.Image] = None,
        # Jev parameter aliases:
        question: Optional[str] = None,
        reference: Optional[str] = None,
        candidate: Optional[str] = None,
    ) -> GradeResult:
        """Grades a candidate answer against a reference answer.
        
        Fully compatible with Jev: returns GradeResult which behaves as a str ("contradiction",
        "entailment", "neutral"), while supporting tuple unpacking (label, prob_dict) and .probabilities.
        """
        query = premise if premise is not None else question
        ref = reference_answer if reference_answer is not None else reference
        cand = candidate_answer if candidate_answer is not None else candidate

        formatted_premise = f"{query}\nReference answer: {ref}"
        formatted_hyp = f"Candidate answer: {cand}"

        pairs = [(formatted_premise, formatted_hyp)]
        images = [image] if image is not None else None

        probs = self.predict(pairs, images)[0]
        label_idx = int(np.argmax(probs))
        label_name = ID2LABEL[label_idx]

        prob_dict = {
            "contradiction": float(probs[CONTRADICTION]),
            "entailment": float(probs[ENTAILMENT]),
            "neutral": float(probs[NEUTRAL]),
        }
        return GradeResult(label_name, prob_dict)


# OpenJEV drop-in alias
OpenJevCrossEncoder = Gemma4CrossEncoder


# -----------------------------------------------------------------------------
# Latent MLP Head (Jev-compatible Soft BCE Downstream Classifier)
# -----------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, d: int, hidden: int = 512, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def soft_bce(logits: torch.Tensor, y: torch.Tensor, eps: float, pos_weight: torch.Tensor) -> torch.Tensor:
    """BCE with soft targets: gold -> 1-eps, others -> eps; positives up-weighted by pos_weight."""
    target = y * (1 - eps) + (1 - y) * eps
    w = torch.where(y > 0.5, pos_weight, torch.ones_like(y))
    return (w * nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")).mean()


def per_question_acc(scores: np.ndarray, qid: np.ndarray, gold: np.ndarray) -> float:
    """Fraction of questions whose argmax-scored option is the gold one."""
    if len(qid) == 0:
        return 0.0
    order = np.argsort(qid, kind="stable")
    scores, qid, gold = scores[order], qid[order], gold[order]
    change_mask = (qid[:-1] != qid[1:]) if len(qid) > 1 else np.array([], dtype=bool)
    starts = np.r_[0, np.flatnonzero(change_mask) + 1, len(qid)]
    hits = [gold[a:b][scores[a:b].argmax()] == 1 for a, b in zip(starts[:-1], starts[1:])]
    return float(np.mean(hits)) if hits else 0.0


def grouped_split(qid: np.ndarray, frac: float, seed: int):
    qs = np.unique(qid)
    if len(qs) <= 1 or frac <= 0.0:
        return np.ones(len(qid), dtype=bool), np.zeros(len(qid), dtype=bool)
    rng = np.random.RandomState(seed)
    rng.shuffle(qs)
    n_val = max(1, min(len(qs) - 1, int(len(qs) * frac)))
    hold = set(qs[:n_val].tolist())
    mask = np.array([q in hold for q in qid])
    return ~mask, mask


class LatentMLPHead:
    """Small MLP head on top of frozen cross-encoder latents trained with soft BCE.
    
    100% drop-in compatible with Jev / openjev LatentMLPHead API.
    """
    def __init__(
        self,
        d: int,
        hidden: int = 512,
        dropout: float = 0.1,
        eps: float = 0.1,
        lr: float = 1e-3,
        wd: float = 1e-2,
        bs: int = 512,
        epochs: int = 60,
        patience: int = 8,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.cfg = dict(d=d, hidden=hidden, dropout=dropout, eps=eps, lr=lr, wd=wd, bs=bs, epochs=epochs, patience=patience, seed=seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.model = _MLP(d, hidden, dropout).to(self.device)
        self.mu, self.sd = np.zeros((1, d), np.float32), np.ones((1, d), np.float32)

    def _t(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.device)

    def fit(self, X: np.ndarray, gold: np.ndarray, qid: np.ndarray, val_frac: float = 0.1):
        """X: (n_pairs, d); gold: 1 for correct option, 0 otherwise; qid: question id grouping."""
        X, gold, qid = np.asarray(X, np.float32), np.asarray(gold, np.float32), np.asarray(qid)
        tr, va = grouped_split(qid, val_frac, self.cfg["seed"])
        self.mu, self.sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        Xtr, ytr = self._t(X[tr]), torch.tensor(gold[tr], device=self.device)
        if np.any(va):
            Xva, yva_qid, yva_gold = self._t(X[va]), qid[va], gold[va]
        else:
            Xva, yva_qid, yva_gold = Xtr, qid[tr], gold[tr]

        p = float(gold[tr].mean())
        pos_weight = torch.tensor((1 - p) / max(p, 1e-6), device=self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        best, best_state, bad = -1.0, None, 0
        for ep in range(self.cfg["epochs"]):
            self.model.train()
            perm = torch.randperm(len(Xtr), device=self.device)
            for s in range(0, len(Xtr), self.cfg["bs"]):
                idx = perm[s : s + self.cfg["bs"]]
                loss = soft_bce(self.model(Xtr[idx]), ytr[idx], self.cfg["eps"], pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                acc = per_question_acc(self.model(Xva).cpu().numpy(), yva_qid, yva_gold)
            if acc > best:
                best, bad, best_state = acc, 0, {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.val_acc = best
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Outputs one logit per pair; pick argmax within each question."""
        self.model.eval()
        return self.model(self._t(np.asarray(X, np.float32))).cpu().numpy()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "head.pt"))
        np.savez(os.path.join(path, "norm.npz"), mu=self.mu, sd=self.sd)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "LatentMLPHead":
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        head = cls(device=device, **cfg)
        head.model.load_state_dict(torch.load(os.path.join(path, "head.pt"), map_location=head.device, weights_only=True))
        z = np.load(os.path.join(path, "norm.npz"))
        head.mu, head.sd = z["mu"], z["sd"]
        head.model.eval()
        return head


# -----------------------------------------------------------------------------
# Quantization-Aware Training (QAT): Multi-Format Extensible Quantization Engine
# -----------------------------------------------------------------------------
FP4_E2M1_GRID = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


class MultiQuantSTE(torch.autograd.Function):
    """Straight-Through Estimator supporting multiple hardware quantization formats:
    - 'nvfp4' / 'fp4': NVIDIA Blackwell E2M1 micro-scaled block quantization
    - 'w4a16' / 'int4': Uniform symmetric INT4 Group-32 (compressed-tensors)
    - 'q4_k_m' / 'q4_k': GGUF k-quant block-256 / sub-block-32 affine quantization
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, quant_format: str = "nvfp4", group_size: int = 32) -> torch.Tensor:
        orig_shape = w.shape
        # Grouping MUST strictly partition the reduction dimension (dim=1 in_features)
        if w.ndim != 2 or w.shape[1] % group_size != 0:
            return w

        fmt = quant_format.lower()
        if fmt in ("nvfp4", "fp4", "fp4_e2m1"):
            # NVIDIA Blackwell FP4 E2M1 with block micro-scaling
            w_grouped = w.view(w.shape[0], -1, group_size)
            max_val = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True) + 1e-8
            scale = max_val / 6.0
            x = w_grouped / scale
            grid = FP4_E2M1_GRID.to(device=w.device, dtype=w.dtype)
            diff = torch.abs(x.unsqueeze(-1) - grid)
            idx = torch.argmin(diff, dim=-1)
            snapped = grid[idx]
            return (snapped * scale).view(orig_shape)

        elif fmt in ("q4_k_m", "q4_k", "k_quant"):
            # GGUF k-quant sub-block affine quantization (32 weights per sub-block)
            w_sub = w.view(w.shape[0], -1, group_size)
            min_val = torch.amin(w_sub, dim=-1, keepdim=True)
            max_val = torch.amax(w_sub, dim=-1, keepdim=True)
            scale = (max_val - min_val) / 15.0 + 1e-8
            q = torch.clamp(torch.round((w_sub - min_val) / scale), 0, 15)
            return (q * scale + min_val).view(orig_shape)

        else:
            # Default: 'w4a16' / 'int4' (INT4 symmetric group-wise)
            w_grouped = w.view(w.shape[0], -1, group_size)
            max_val = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True) + 1e-8
            scale = max_val / 7.0
            w_int = torch.clamp(torch.round(w_grouped / scale), -8, 7)
            return (w_int * scale).view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-Through Estimator (STE): pass gradients directly through discretization
        return grad_output, None, None


# Backward-compatible alias
FakeQuantizeSTE = MultiQuantSTE


class FakeQuantizeParametrization(nn.Module):
    """PyTorch parametrization that injects MultiQuantSTE into nn.Linear weights."""

    def __init__(self, quant_format: str = "nvfp4", group_size: int = 32, num_bits: int = 4):
        super().__init__()
        self.quant_format = quant_format
        self.group_size = group_size
        self.num_bits = num_bits

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return MultiQuantSTE.apply(w, self.quant_format, self.group_size)


def apply_quantization_aware_training(
    model: nn.Module,
    quant_format: str = "nvfp4",
    num_bits: int = 4,
    group_size: int = 32,
    target_substrs: Sequence[str] = ("self_attn", "mlp"),
    preserve_mqa: bool = True,
) -> nn.Module:
    """Injects hardware-accurate fake-quantization with Straight-Through Estimator into target linear layers.

    Works seamlessly with LoRA: fake-quantizes the base layer so that adapter weights
    learn directly to cancel out and compensate for target quantization error.

    Supported Formats:
    - 'nvfp4': NVIDIA Blackwell FP4 E2M1 micro-scaled quantization (peak RTX 5090 speed).
    - 'q4_k_m': GGUF k-quant block-256 / sub-block-32 affine quantization.
    - 'w4a16': Standard INT4 symmetric group-32 (compressed-tensors layout).

    Hardened Rules:
    1. Strictly whitelists text decoder projections (language_model.layers).
    2. Vision tower (SigLIP ViT) and multimodal projection adapters are NEVER quantized.
    3. Preserves MQA: In Gemma 4 E2B (num_kv_heads == 1), k_proj and v_proj are kept in 16-bit
       to prevent 8x concentrated quantization noise on the single key/value channel.
    4. Grouping is strictly aligned along in_features (dim=1).
    """
    injected_count = 0
    skipped_count = 0

    for name, module in model.named_modules():
        # Strictly whitelist language model layers and forbid vision tower, classification head, or LoRA adapters
        if "vision_tower" in name or "embed_vision" in name or "score" in name or "norm" in name or "lora" in name:
            continue
        if "layers" not in name:
            continue

        # MQA protection: Single KV head in E2B serves 8 query heads
        if preserve_mqa and any(k in name for k in ("k_proj", "v_proj")):
            skipped_count += 1
            continue

        target = getattr(module, "base_layer", module)
        if isinstance(target, nn.Linear) and any(s in name for s in target_substrs):
            # Avoid re-registering if already parametrized
            if not hasattr(target, "parametrizations") or "weight" not in target.parametrizations:
                # Ensure in_features dimension is divisible by group_size
                if target.weight.shape[1] % group_size == 0:
                    parametrize.register_parametrization(
                        target,
                        "weight",
                        FakeQuantizeParametrization(quant_format=quant_format, group_size=group_size, num_bits=num_bits),
                    )
                    injected_count += 1

    print(
        f"[MultiQuant QAT Engine] Injected '{quant_format}' QAT (group_size={group_size}) into {injected_count} linear layers "
        f"(skipped {skipped_count} sensitive MQA projections)."
    )
    return model



if __name__ == "__main__":
    print("=" * 60)
    print("Running Gemma 4 Cross-Encoder Module Self-Test")
    print("=" * 60)

    from transformers import Gemma4TextConfig, Gemma4VisionConfig

    # 1. Instantiate minimal configuration for verification
    text_cfg = Gemma4TextConfig(
        vocab_size=5000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        sliding_window=64,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=5000,
        layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=0,
    )
    vision_cfg = Gemma4VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        patch_size=16,
        pooling_kernel_size=2,
        position_embedding_size=256,
    )
    full_cfg = Gemma4Config(
        text_config=text_cfg,
        vision_config=vision_cfg,
        boi_token_id=4990,
        eoi_token_id=4991,
        image_token_id=4992,
        num_labels=3,
    )

    print("Initializing Gemma4ForSequenceClassification...")
    model = Gemma4ForSequenceClassification(full_cfg)
    print("  Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    # 2. Test freeze vision tower
    model.freeze_vision_tower(freeze_adapter=False)
    vis_frozen = all(not p.requires_grad for p in model.model.vision_tower.parameters())
    print("  Vision tower frozen:", vis_frozen)

    # 3. Test text-only forward pass & loss
    input_ids = torch.tensor([[10, 20, 30, 0], [40, 50, 60, 70]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    labels = torch.tensor([ENTAILMENT, CONTRADICTION])

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print("  Text logits shape:", tuple(out.logits.shape))
    print("  Text loss:", f"{out.loss.item():.4f}")
    assert out.logits.shape == (2, 3), "Logits shape mismatch"

    # Backward gradient test
    out.loss.backward()
    print("  Backward gradient pass: SUCCESS")

    print("\nGemma 4 Cross-Encoder Module Verified Successfully!")
    print("=" * 60)

