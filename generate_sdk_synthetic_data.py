#!/usr/bin/env python3
"""generate_sdk_synthetic_data.py - Synthetic Data Generation & Cross-Family Validation Engine.

Aligns the NLI cross-encoder training curriculum with the EXACT interaction patterns
exposed in the user-facing SDK (gemma4_cross_encoder.py):
1. `ce.rerank(query, documents, scoring=...)`: Search & passage reranking with BM25-style hard distractors.
2. `ce.grade(questions, responses, threshold=...)`: LLM-as-a-judge rubric & reference answer grading.
3. Zero-shot Tool & Intent Routing: Mapping user utterances to tool schemas without text generation.
4. Multiple-Choice Cloze Decision: ARC/MMLU/WinoGrande style stem-to-candidate option ranking.
5. Subtle Enterprise RAG Hallucination Detection: Document context vs nuanced claims (entity/date/causal swaps).
6. Structured / Multimodal Table & UI State Verification: Checking structured layouts and data tables.

Features:
- Four-Judge Cross-Family Consensus: teacher (Gemma 4 31B) + validators (Qwen 3.6 27B,
  DeepSeek V4 Flash, Qwen 3.8 125B q4/q3). Judges run one at a time over ALL batches
  (llama-swap serves --models-max 1, so per-batch switching would thrash reloads).
- Disagreement Review Queue: non-unanimous committees, judge failures and unanimous
  overrides land in sdk_synthetic_disagreements.jsonl with review.status="pending"
  for human / cloud-LLM adjudication.
- Judge Metrics DB: every verdict & batch persists to SQLite (validation_metrics.db,
  `judge_performance` view) for per-judge accuracy/failure/latency analysis.
- GBNF Token Restriction: strict JSON schemas are compiled to GBNF grammars by llama-server.
- Offline High-Fidelity Generative Fallback: Includes domain templates and perturbation heuristics so it can run standalone.
- Rejection Sampling & Label Disambiguation: Eliminates the fatal "Neutral vs Contradiction" synthetic failure mode.
"""

import os
import re
import json
import time
import random
import argparse
from typing import Dict, List, Any, Optional, Tuple

# Shared domain constants & transport live in dedicated modules now:
# - nli_labels.py: label enum (re-exported here for backward compatibility)
# - llm_client.py: LLMEndpointClient (re-exported here for backward compatibility)
# - validator_committee.py: 4-judge consensus, disagreement queue & metrics DB
from nli_labels import CONTRADICTION, ENTAILMENT, ID2LABEL, LABEL2ID, NEUTRAL  # noqa: F401
from llm_client import LLMEndpointClient  # noqa: F401
from validation_metrics_db import RunSpec, ValidationMetricsDB
from validator_committee import (
    AggregateResult,
    aggregate_committee_votes,
    print_judge_summary,
    run_validator_committee,
    write_disagreement_queue,
)


TEACHER_DOMAIN_TOPICS = [
    ("cloud_kubernetes", "Kubernetes cluster scaling, pod eviction, and ingress controller routing."),
    ("financial_fraud", "Real-time payment transaction monitoring, AML compliance, and credit risk limits."),
    ("pharmacovigilance", "Adverse drug reaction reporting, clinical trial dosage protocols, and contraindication checking."),
    ("database_replication", "Distributed consensus, write-ahead logging, and multi-region database replication."),
    ("enterprise_security", "Zero-trust IAM policy evaluation, role-based access control, and vulnerability patching."),
]

NLI_TRIPLE_ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "premise": {"type": "string"},
            "hypothesis": {"type": "string"},
            "label": {"type": "integer", "enum": [0, 1, 2]},
            "metadata": {"type": "object"}
        },
        "required": ["premise", "hypothesis", "label"]
    },
    "minItems": 3,
    "maxItems": 3
}


def generate_teacher_domain_samples(
    teacher_client: LLMEndpointClient,
    topics: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Prompts the teacher LLM (using optimized alias e.g. gemma-4-31b-q4) to generate high-complexity domain NLI triples."""
    if topics is None:
        topics = TEACHER_DOMAIN_TOPICS

    results: List[Dict[str, Any]] = []
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "nli_triples",
            "strict": True,
            "schema": NLI_TRIPLE_ARRAY_SCHEMA
        }
    }

    for topic_id, topic_desc in topics:
        print(f"  Querying teacher ({teacher_client.model}) for topic: {topic_id}...")
        system_prompt = (
            "You are an expert training data generator for NLI cross-encoders and System 1 decision routing. "
            "Think through subtle semantic nuances, domain edge cases, and factual boundary conditions."
        )
        user_prompt = (
            f"Generate exactly 3 samples for the domain: {topic_desc}. "
            f"Provide 1 Entailment (label 1), 1 Contradiction (label 0), and 1 Neutral (label 2)."
        )

        resp = teacher_client.query_chat(
            system_prompt,
            user_prompt,
            temperature=0.4,
            max_tokens=1500,
            response_format=response_format,
        )
        if not resp:
            continue

        try:
            items = json.loads(resp)
            for it in items:
                results.append({
                    "id": f"teacher_{teacher_client.model}_{len(results):06d}",
                    "premise": it["premise"],
                    "hypothesis": it["hypothesis"],
                    "label": int(it["label"]),
                    "source": f"teacher_{topic_id}",
                    "language": "en",
                    "image": "",
                    "metadata": it.get("metadata", {"domain": topic_id}),
                })
        except Exception as e:
            print(f"  [Parse fallback for {topic_id}] {e}")
            m = re.search(r"\[.*\]", resp, re.DOTALL)
            if m:
                try:
                    items = json.loads(m.group(0))
                    for it in items:
                        results.append({
                            "id": f"teacher_{teacher_client.model}_{len(results):06d}",
                            "premise": it["premise"],
                            "hypothesis": it["hypothesis"],
                            "label": int(it["label"]),
                            "source": f"teacher_{topic_id}",
                            "language": "en",
                            "image": "",
                            "metadata": it.get("metadata", {"domain": topic_id}),
                        })
                except Exception:
                    pass

    return results


# -----------------------------------------------------------------------------
# 1. Tool & Intent Routing Generator (ce.rerank for Tool Dispatch)
# -----------------------------------------------------------------------------
TOOL_REGISTRY = [
    {
        "name": "search_code_graph",
        "description": "Search the codebase knowledge graph for functions, classes, routes, and symbol definitions.",
        "domains": ["code", "development", "refactoring"],
        "entailment_queries": [
            "Where is the Gemma4CrossEncoder class defined in our project?",
            "Find the function that unpacks INT4 weights in export_w4a16.py",
            "Show me the call hierarchy of evaluate_zero_shot_tool_routing",
            "Search the repository for references to FakeQuantizeSTE",
            "Locate all methods associated with sequence classification pooling",
        ],
        "contradiction_queries": [
            "Send an email to the client with the quarterly marketing report.",
            "Transfer $500 from checking to savings account.",
            "Generate a photorealistic image of a sunset over the ocean.",
            "Book a non-stop flight from San Francisco to Tokyo next Tuesday.",
            "Adjust the smart thermostat in the living room to 72 degrees.",
        ],
        "neutral_queries": [
            "What is the best way to design a software architecture?",
            "Why is Python popular for machine learning?",
            "How does a knowledge graph work in theoretical computer science?",
            "Can you write a poem about software engineering?",
            "I need some advice on choosing between C++ and Rust.",
        ],
    },
    {
        "name": "weather_forecast_service",
        "description": "Retrieve current meteorological data, hourly forecasts, and precipitation alerts for a city or coordinates.",
        "domains": ["weather", "daily", "travel"],
        "entailment_queries": [
            "Will it rain in Seattle tomorrow afternoon?",
            "What is the current temperature and humidity in Tokyo?",
            "Check if there are any severe thunderstorm warnings for Miami tonight.",
            "Do I need an umbrella for my morning commute in London?",
            "Show the 7-day weather forecast for Denver, Colorado.",
        ],
        "contradiction_queries": [
            "Git push the current branch to origin main with force.",
            "Calculate the determinant of a 4x4 covariance matrix.",
            "Translate this legal contract from German to Japanese.",
            "Order a pepperoni pizza from Domino's with extra cheese.",
            "Summarize the Q3 earnings transcript for Nvidia.",
        ],
        "neutral_queries": [
            "Why does it rain so much in the Pacific Northwest?",
            "Explain the Coriolis effect and how hurricanes form.",
            "What was the weather like on the day the Titanic sank in 1912?",
            "I love sunny days when the sky is completely clear.",
            "How do meteorologists build numerical weather prediction models?",
        ],
    },
    {
        "name": "financial_transaction_gateway",
        "description": "Execute wire transfers, peer-to-peer payments, ACH transfers, and check bank balance.",
        "domains": ["banking", "finance", "payments"],
        "entailment_queries": [
            "Send $120 to Alice for dinner expenses.",
            "Transfer $1,000 from my primary checking to the high-yield savings account.",
            "Pay my outstanding credit card balance of $452.18 immediately.",
            "Initiate a wire transfer to escrow account 982341 for the house closing.",
            "What is my current available checking balance after yesterday's payroll?",
        ],
        "contradiction_queries": [
            "Resize this JPEG image to 1024x1024 without cropping.",
            "What is the speed of light in a vacuum?",
            "Debug why my Docker container is failing to mount the volume.",
            "Find research papers on diffusion transformers published in 2026.",
            "Play the latest album by Daft Punk on the bedroom speaker.",
        ],
        "neutral_queries": [
            "What is the historical average annual return of the S&P 500?",
            "How does inflation affect the purchasing power of fiat currency?",
            "Should I invest in index funds or individual equities?",
            "Explain how fractional-reserve banking systems operate.",
            "What is the difference between a debit card and a credit card?",
        ],
    },
    {
        "name": "image_generation_service",
        "description": "Synthesize or edit images, illustrations, and 3D mockups from natural language prompts.",
        "domains": ["vision", "creative", "media"],
        "entailment_queries": [
            "Generate a watercolor painting of an astronaut drinking coffee on Mars.",
            "Render a modern minimalist logo for a cybersecurity startup featuring an owl.",
            "Create a photorealistic 3D render of a futuristic sports car in neon cyberpunk lighting.",
            "Draw a vector icon of a magnifying glass over a neural network.",
            "Produce an oil painting portrait of a cat dressed in Victorian aristocratic attire.",
        ],
        "contradiction_queries": [
            "Run unit tests across the entire Python test suite with pytest.",
            "What is the capital city of Madagascar?",
            "Cancel my subscription to the Wall Street Journal.",
            "Calculate the integral of x^2 * sin(x) dx.",
            "Check my calendar for meetings scheduled after 3 PM today.",
        ],
        "neutral_queries": [
            "Who painted the Mona Lisa and in what year was it completed?",
            "What artistic movement did Salvador Dali belong to?",
            "Explain the difference between vector graphics and raster images.",
            "Why is lighting so important in photography and visual art?",
            "Can you critique the composition of classic Renaissance paintings?",
        ],
    },
    {
        "name": "web_search_index",
        "description": "Query the live public web for current news, articles, documentation, and external facts.",
        "domains": ["search", "web", "live_data"],
        "entailment_queries": [
            "Who won the gold medal in the men's 100m sprint yesterday?",
            "What are the latest updates on the Mars sample return mission announced this week?",
            "Search for recent tech industry news regarding RTX 5090 availability.",
            "Find the official documentation for Google Gemma 4 architecture released in 2026.",
            "What was the closing stock price of Apple today?",
        ],
        "contradiction_queries": [
            "Turn off the lights in the master bedroom.",
            "Sort this array of numbers in descending order: [5, 2, 9, 1].",
            "Mute my microphone during the ongoing Zoom call.",
            "Generate an INT4 packed safetensors file for our cross-encoder.",
            "Draft a polite apology letter to my landlord for late rent.",
        ],
        "neutral_queries": [
            "How did Tim Berners-Lee invent the World Wide Web at CERN?",
            "What is the difference between HTTP and HTTPS protocols?",
            "Why do web browsers use caching to speed up page loads?",
            "Explain how search engines index billions of web pages efficiently.",
            "Is the internet an essential public utility in the modern world?",
        ],
    },
    {
        "name": "calendar_and_scheduling",
        "description": "Manage calendar events, schedule appointments, send meeting invitations, and check availability.",
        "domains": ["productivity", "calendar", "meetings"],
        "entailment_queries": [
            "Schedule a 30-minute sync with Dave for tomorrow at 2 PM.",
            "Set a calendar reminder for my dentist appointment next Friday at 9 AM.",
            "Move our project review meeting from Thursday to Friday afternoon.",
            "What meetings do I have scheduled on my calendar for today?",
            "Invite Sarah and John to the quarterly planning session next Monday.",
        ],
        "contradiction_queries": [
            "Convert 100 degrees Fahrenheit to Celsius.",
            "Compress this video file using the H.265 codec at 1080p.",
            "What is the chemical formula for photosynthesis?",
            "Deploy the docker image to our Kubernetes production cluster.",
            "Write a regex to match valid IPv4 addresses.",
        ],
        "neutral_queries": [
            "Why do different cultures use different calendar systems throughout history?",
            "What is the difference between the Julian and Gregorian calendars?",
            "How can one maintain good time management skills when working remotely?",
            "Explain why leap years exist every four years.",
            "I often feel like there aren't enough hours in a working day.",
        ],
    },
]


TOOL_HYPOTHESIS_TEMPLATES = [
    "The appropriate tool to handle this request is: {}",
    "Tool candidate: {}",
    "Dispatch request to: {}",
    "Target system: {}",
    "Routing destination: {}",
    "{}",
]


def generate_tool_routing_samples(n_target: int = 1500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates pairs matching SDK tool routing with template diversification
    and symmetric polarity balance (order/prefix invariance).
    Premise: 'User request: {query}'
    Hypothesis: '{template.format(tool_description)}'
    """
    rng = random.Random(seed)
    samples = []
    sample_id = 0

    while len(samples) < n_target:
        tool = rng.choice(TOOL_REGISTRY)
        name = tool["name"]
        desc = tool["description"]
        fmt = rng.choice(TOOL_HYPOTHESIS_TEMPLATES)

        # 1. Entailment
        q_ent = rng.choice(tool["entailment_queries"])
        # Symmetrized polarity (20% negative assertion)
        if rng.random() < 0.2:
            samples.append({
                "id": f"sdk_tool_{sample_id:06d}",
                "premise": f"User request: {q_ent}",
                "hypothesis": f"This request should NOT be handled by: {desc}",
                "label": CONTRADICTION,
                "source": "sdk_tool_routing",
                "language": "en",
                "image": "",
                "metadata": {"tool": name, "intent": "positive_inverted_polarity"},
            })
        else:
            samples.append({
                "id": f"sdk_tool_{sample_id:06d}",
                "premise": f"User request: {q_ent}",
                "hypothesis": fmt.format(desc),
                "label": ENTAILMENT,
                "source": "sdk_tool_routing",
                "language": "en",
                "image": "",
                "metadata": {"tool": name, "intent": "positive"},
            })
        sample_id += 1

        # 2. Contradiction
        q_con = rng.choice(tool["contradiction_queries"])
        if rng.random() < 0.2:
            samples.append({
                "id": f"sdk_tool_{sample_id:06d}",
                "premise": f"User request: {q_con}",
                "hypothesis": f"This request should NOT be handled by: {desc}",
                "label": ENTAILMENT,
                "source": "sdk_tool_routing",
                "language": "en",
                "image": "",
                "metadata": {"tool": name, "intent": "negative_inverted_polarity"},
            })
        else:
            samples.append({
                "id": f"sdk_tool_{sample_id:06d}",
                "premise": f"User request: {q_con}",
                "hypothesis": fmt.format(desc),
                "label": CONTRADICTION,
                "source": "sdk_tool_routing",
                "language": "en",
                "image": "",
                "metadata": {"tool": name, "intent": "conflicting"},
            })
        sample_id += 1

        # 3. Neutral (Ambiguous or tangential)
        q_neu = rng.choice(tool["neutral_queries"])
        samples.append({
            "id": f"sdk_tool_{sample_id:06d}",
            "premise": f"User request: {q_neu}",
            "hypothesis": fmt.format(desc),
            "label": NEUTRAL,
            "source": "sdk_tool_routing",
            "language": "en",
            "image": "",
            "metadata": {"tool": name, "intent": "ambiguous"},
        })
        sample_id += 1

    rng.shuffle(samples)
    return samples[:n_target]



# -----------------------------------------------------------------------------
# 2. Response & Rubric Grading Generator (ce.grade)
# -----------------------------------------------------------------------------
RUBRIC_TEMPLATES = [
    {
        "question": "What is the primary function of mitochondria in eukaryotic cells?",
        "reference": "Mitochondria generate most of the cell's ATP through cellular respiration and oxidative phosphorylation, functioning as the powerhouse of the cell.",
        "entailment_candidates": [
            "Mitochondria are responsible for producing the vast majority of cellular ATP via oxidative phosphorylation during aerobic respiration.",
            "Their main role is to act as cellular powerhouses, synthesizing ATP to fuel cellular processes through cellular respiration.",
            "They produce chemical energy in the form of ATP by carrying out oxidative phosphorylation.",
        ],
        "contradiction_candidates": [
            "Mitochondria are responsible for photosynthesis and the synthesis of glucose using sunlight.",
            "Their primary function is to store genetic material in the cytoplasm and synthesize ribosomes.",
            "Mitochondria break down ATP into glucose to cool the cell down during anaerobic metabolism.",
        ],
        "neutral_candidates": [
            "Mitochondria have their own circular mitochondrial DNA and are believed to have evolved via endosymbiosis.",
            "They possess both an inner membrane and an outer membrane with folding known as cristae.",
            "Many human diseases can be caused by mutations inherited through maternal mitochondrial genes.",
        ],
    },
    {
        "question": "How does gradient descent update parameters during neural network training?",
        "reference": "Gradient descent computes the gradient of the loss function with respect to each parameter, then updates each parameter in the opposite direction of the gradient scaled by the learning rate.",
        "entailment_candidates": [
            "Parameters are adjusted by calculating the loss gradient and taking a step in the negative gradient direction multiplied by the learning rate.",
            "It evaluates the partial derivatives of the loss and subtracts the product of the learning rate and gradient from the current weights.",
            "By computing gradients via backpropagation and moving weights in the opposite direction of steepest ascent proportional to the learning rate.",
        ],
        "contradiction_candidates": [
            "Gradient descent increases parameters along the direction of steepest ascent to maximize the training loss.",
            "It randomly shuffles network weights without using derivatives until the accuracy reaches a threshold.",
            "Parameters are updated by dividing the weights by the gradient regardless of the loss surface.",
        ],
        "neutral_candidates": [
            "Common variants of gradient descent include Adam, RMSProp, and AdamW with momentum.",
            "Stochastic gradient descent was first introduced by Robbins and Monro in 1951.",
            "Choosing an appropriate learning rate is crucial to avoid gradient explosion or vanishing gradients.",
        ],
    },
    {
        "question": "What causes the change of seasons on Earth?",
        "reference": "Earth's seasons are caused by the 23.5-degree tilt of Earth's rotational axis relative to its orbital plane as it orbits the Sun.",
        "entailment_candidates": [
            "Seasons occur because Earth's axis of rotation is tilted at approximately 23.5 degrees as it revolves around the Sun.",
            "The axial tilt of the Earth causes different hemispheres to receive varying angles of sunlight throughout its annual orbit.",
            "Earth's 23.5-degree axial tilt alters solar irradiance across hemispheres during the yearly revolution.",
        ],
        "contradiction_candidates": [
            "Seasons are caused by Earth moving substantially closer to the Sun in the summer and farther away in the winter.",
            "The Moon's gravitational pull slows down Earth's rotation, resulting in seasonal temperature fluctuations.",
            "Seasonal changes occur because the Sun burns hotter during July and cooler during January.",
        ],
        "neutral_candidates": [
            "The summer solstice represents the longest day of daylight in the Northern Hemisphere.",
            "Equinoxes happen twice a year when day and night are approximately equal in duration across the globe.",
            "Other planets in the solar system, such as Mars, also experience seasons due to their axial tilt.",
        ],
    },
    {
        "question": "Explain the difference between synchronous and asynchronous I/O in programming.",
        "reference": "Synchronous I/O blocks thread execution until the I/O operation finishes, while asynchronous I/O allows the thread to continue executing other tasks while the I/O runs in the background, notifying completion via callbacks, promises, or an event loop.",
        "entailment_candidates": [
            "In synchronous I/O, the calling process waits and halts until the operation completes; in asynchronous I/O, execution continues non-blocking while the operation proceeds.",
            "Synchronous operations block the execution thread until data is returned, whereas asynchronous calls return immediately and notify completion via an event loop or promise.",
            "Blocking synchronous I/O halts the caller, while asynchronous I/O operates concurrently without stalling the main execution flow.",
        ],
        "contradiction_candidates": [
            "Synchronous I/O never waits for operations to complete, whereas asynchronous I/O halts the entire operating system until all threads finish.",
            "Asynchronous I/O requires multiple CPU cores and cannot run on single-threaded event loops like Node.js.",
            "Synchronous I/O is always faster and consumes less memory because it runs entirely on the GPU.",
        ],
        "neutral_candidates": [
            "Python introduced native async/await keywords in version 3.5 with the asyncio module.",
            "Network sockets and disk file descriptors are standard abstractions provided by POSIX operating systems.",
            "Epoll and kqueue are scalable I/O event notification mechanisms used by modern web servers like Nginx.",
        ],
    },
    {
        "question": "What is the role of the Federal Reserve in managing US monetary policy?",
        "reference": "The Federal Reserve manages US monetary policy to promote maximum employment and stable prices (the dual mandate) primarily by setting the federal funds rate and adjusting the money supply.",
        "entailment_candidates": [
            "The Fed implements monetary policy to achieve maximum employment and price stability by manipulating the federal funds interest rate.",
            "Its mandate is dual: maintaining price stability and maximizing employment, which it pursues through interest rate decisions and open market operations.",
            "The Federal Reserve guides economic stability and inflation targets by setting benchmark interest rates and managing systemic liquidity.",
        ],
        "contradiction_candidates": [
            "The Federal Reserve sets income tax brackets and directly decides federal government spending budgets.",
            "The Fed is a branch of the US military responsible for protecting gold bullion stored at Fort Knox.",
            "The Federal Reserve guarantees that no company in the United States can ever file for bankruptcy.",
        ],
        "neutral_candidates": [
            "The Federal Reserve was established by the Federal Reserve Act signed by President Woodrow Wilson in 1913.",
            "The Federal Open Market Committee (FOMC) holds eight regularly scheduled meetings each year.",
            "Jerome Powell was appointed Chairman of the Board of Governors of the Federal Reserve System.",
        ],
    },
]


def generate_rubric_grading_samples(n_target: int = 1500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates pairs matching SDK ce.grade():
    Premise: '{question}\\nReference answer: {reference}'
    Hypothesis: 'Candidate answer: {candidate}'
    """
    rng = random.Random(seed)
    samples = []
    sample_id = 0

    while len(samples) < n_target:
        item = rng.choice(RUBRIC_TEMPLATES)
        q = item["question"]
        ref = item["reference"]
        formatted_premise = f"{q}\nReference answer: {ref}"

        # 1. Entailment (Grounded correct answer)
        cand_ent = rng.choice(item["entailment_candidates"])
        samples.append({
            "id": f"sdk_grade_{sample_id:06d}",
            "premise": formatted_premise,
            "hypothesis": f"Candidate answer: {cand_ent}",
            "label": ENTAILMENT,
            "source": "sdk_rubric_grading",
            "language": "en",
            "image": "",
            "metadata": {"question": q, "status": "factual_pass"},
        })
        sample_id += 1

        # 2. Contradiction (Factual error / violation)
        cand_con = rng.choice(item["contradiction_candidates"])
        samples.append({
            "id": f"sdk_grade_{sample_id:06d}",
            "premise": formatted_premise,
            "hypothesis": f"Candidate answer: {cand_con}",
            "label": CONTRADICTION,
            "source": "sdk_rubric_grading",
            "language": "en",
            "image": "",
            "metadata": {"question": q, "status": "factual_fail"},
        })
        sample_id += 1

        # 3. Neutral (Tangential / unverified claim)
        cand_neu = rng.choice(item["neutral_candidates"])
        samples.append({
            "id": f"sdk_grade_{sample_id:06d}",
            "premise": formatted_premise,
            "hypothesis": f"Candidate answer: {cand_neu}",
            "label": NEUTRAL,
            "source": "sdk_rubric_grading",
            "language": "en",
            "image": "",
            "metadata": {"question": q, "status": "tangential"},
        })
        sample_id += 1

    rng.shuffle(samples)
    return samples[:n_target]


# -----------------------------------------------------------------------------
# 3. Search Reranking & Retrieval with BM25 Hard Distractors (ce.rerank)
# -----------------------------------------------------------------------------
SEARCH_RERANK_TOPICS = [
    {
        "query": "What is the optimal batch size for fine-tuning LoRA adapters on RTX 5090?",
        "positive": "For an RTX 5090 with 32 GB VRAM, fine-tuning LoRA with rank 64 on a 2B parameter backbone typically uses a batch size of 16 to 32 with gradient accumulation of 2 to avoid memory fragmentation.",
        "hard_negative_neutral": "The RTX 5090 features 32,607 MiB of GDDR7 VRAM and Blackwell architecture compute cores, offering over 3,000 TFLOPS of FP4 tensor performance.",
        "contradiction": "An RTX 5090 only has 8 GB of VRAM, so LoRA fine-tuning must be restricted to a batch size of 1 with all layers frozen to avoid immediate out-of-memory crashes.",
    },
    {
        "query": "How does FlashAttention-3 reduce memory bandwidth bottlenecks on Hopper and Blackwell?",
        "positive": "FlashAttention-3 leverages asynchronous tensor core operations (TMA), warpgroup-level matrix multiplies, and FP8 precision to overlap softmax computation with memory loads.",
        "hard_negative_neutral": "Attention mechanisms in transformer models originally scaled quadratically with sequence length O(N^2) before tiling techniques were introduced by Dao et al.",
        "contradiction": "FlashAttention-3 eliminates the attention matrix entirely by replacing self-attention with state-space recurrence layers that require zero memory bandwidth.",
    },
    {
        "query": "Why should last non-pad token pooling be used instead of mean pooling for decoder LLM cross-encoders?",
        "positive": "In causal decoder-only models, each token can only attend to previous tokens via causal masks, meaning the final non-pad token has aggregated contextual information across the entire sequence.",
        "hard_negative_neutral": "Mean pooling calculates the arithmetic average of all token representations in the sequence length dimension, commonly used in bidirectional encoders like BERT and ModernBERT.",
        "contradiction": "Last token pooling is strictly forbidden in decoder models because causal masks prevent the last token from attending to earlier premise tokens.",
    },
    {
        "query": "What is the purpose of straight-through estimators in quantization-aware training?",
        "positive": "The Straight-Through Estimator (STE) rounds weights to discrete integer levels during the forward pass while copying the incoming gradients unchanged during the backward pass, allowing backpropagation through non-differentiable rounding.",
        "hard_negative_neutral": "Post-training quantization (PTQ) calibrates scale factors without backpropagation, whereas QAT fine-tunes weights in the presence of simulated quantization noise.",
        "contradiction": "Straight-through estimators calculate the exact mathematical analytical derivative of the discontinuous step function, producing zero gradients everywhere during backpropagation.",
    },
    {
        "query": "How does log-odds margin scoring prevent neutral prior collapse during zero-shot reranking?",
        "positive": "Log-odds margin scoring computes z_entailment - z_contradiction, which algebraically subtracts out the neutral logit and prevents candidate options without context from being penalized by high neutral probability.",
        "hard_negative_neutral": "The Softmax function normalizes logits across classes such that the sum of probabilities equals 1.0, preserving the relative ranking of positive inputs.",
        "contradiction": "Log-odds margin scoring forces all neutral probabilities to 100%, ensuring that only neutral classes can win the argmax decision during reranking.",
    },
]


def generate_search_rerank_samples(n_target: int = 1500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates pairs matching SDK search reranking:
    Premise: '{query}'
    Hypothesis: '{candidate_passage}'
    """
    rng = random.Random(seed)
    samples = []
    sample_id = 0

    while len(samples) < n_target:
        topic = rng.choice(SEARCH_RERANK_TOPICS)
        q = topic["query"]

        # 1. Entailment (Relevant answering passage)
        samples.append({
            "id": f"sdk_rerank_{sample_id:06d}",
            "premise": q,
            "hypothesis": topic["positive"],
            "label": ENTAILMENT,
            "source": "sdk_search_reranking",
            "language": "en",
            "image": "",
            "metadata": {"query": q, "type": "gold_answer"},
        })
        sample_id += 1

        # 2. Contradiction (Factual corruption of answer)
        samples.append({
            "id": f"sdk_rerank_{sample_id:06d}",
            "premise": q,
            "hypothesis": topic["contradiction"],
            "label": CONTRADICTION,
            "source": "sdk_search_reranking",
            "language": "en",
            "image": "",
            "metadata": {"query": q, "type": "inverted_answer"},
        })
        sample_id += 1

        # 3. Neutral (BM25 lexical overlap distractor - related topic but does not answer)
        samples.append({
            "id": f"sdk_rerank_{sample_id:06d}",
            "premise": q,
            "hypothesis": topic["hard_negative_neutral"],
            "label": NEUTRAL,
            "source": "sdk_search_reranking",
            "language": "en",
            "image": "",
            "metadata": {"query": q, "type": "lexical_distractor"},
        })
        sample_id += 1

    rng.shuffle(samples)
    return samples[:n_target]


# -----------------------------------------------------------------------------
# 4. Multiple-Choice Cloze / ARC-Style Decision Generator
# -----------------------------------------------------------------------------
CLOZE_TASKS = [
    {
        "stem": "Which property of water is most responsible for its ability to dissolve a wide variety of solutes?",
        "correct": "Water molecules are polar and form hydrogen bonds with solutes.",
        "distractor_neutral_1": "Water covers more than seventy percent of the surface of the Earth.",
        "distractor_neutral_2": "Water has a relatively high boiling point compared to hydrogen sulfide.",
        "contradiction": "Water is a completely nonpolar substance that repels all ionic compounds.",
    },
    {
        "stem": "In the human circulatory system, deoxygenated blood returns from the body tissues into which chamber of the heart?",
        "correct": "Right atrium",
        "distractor_neutral_1": "Left ventricle, which pumps blood into the systemic aorta.",
        "distractor_neutral_2": "Pulmonary vein, which carries oxygenated blood from the lungs.",
        "contradiction": "Left atrium, which directly receives oxygen-depleted blood from the vena cava.",
    },
    {
        "stem": "Which layer of Earth's atmosphere contains the majority of the ozone layer that absorbs harmful ultraviolet radiation?",
        "correct": "Stratosphere",
        "distractor_neutral_1": "Troposphere, where almost all weather phenomena take place.",
        "distractor_neutral_2": "Thermosphere, where auroras and high-energy ionizations occur.",
        "contradiction": "The ozone layer is entirely located within Earth's molten liquid outer core.",
    },
    {
        "stem": "In computer science, what is the asymptotic worst-case time complexity of searching for an element in a balanced binary search tree with N nodes?",
        "correct": "O(log N)",
        "distractor_neutral_1": "O(N log N), which is the optimal comparison-based sorting complexity.",
        "distractor_neutral_2": "O(1), which is the average lookup time in an ideal hash table.",
        "contradiction": "O(2^N), because balanced binary search trees require exponential time for basic traversal.",
    },
]


CLOZE_HYPOTHESIS_TEMPLATES = [
    "The correct answer is: {}",
    "Candidate option: {}",
    "Statement: {}",
    "Claim: {}",
    "Hypothesis: {}",
    "{}",
    "Option: {}",
    "Selected answer: {}",
]


def generate_cloze_decision_samples(n_target: int = 1500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates pairs matching SDK multiple-choice reranking with hypothesis template diversification:
    Premise: '{stem}'
    Hypothesis: '{template.format(option)}'
    """
    rng = random.Random(seed)
    samples = []
    sample_id = 0

    while len(samples) < n_target:
        task = rng.choice(CLOZE_TASKS)
        stem = task["stem"]
        fmt = rng.choice(CLOZE_HYPOTHESIS_TEMPLATES)

        # 1. Entailment (Correct answer)
        samples.append({
            "id": f"sdk_cloze_{sample_id:06d}",
            "premise": stem,
            "hypothesis": fmt.format(task["correct"]),
            "label": ENTAILMENT,
            "source": "sdk_cloze_reasoning",
            "language": "en",
            "image": "",
            "metadata": {"stem": stem, "choice": "correct"},
        })
        sample_id += 1

        # 2. Contradiction (Blatantly false choice)
        samples.append({
            "id": f"sdk_cloze_{sample_id:06d}",
            "premise": stem,
            "hypothesis": fmt.format(task["contradiction"]),
            "label": CONTRADICTION,
            "source": "sdk_cloze_reasoning",
            "language": "en",
            "image": "",
            "metadata": {"stem": stem, "choice": "contradiction"},
        })
        sample_id += 1

        # 3. Neutral (Plausible distractor: true fact, but does not answer the question)
        d_neu = rng.choice([task["distractor_neutral_1"], task["distractor_neutral_2"]])
        samples.append({
            "id": f"sdk_cloze_{sample_id:06d}",
            "premise": stem,
            "hypothesis": fmt.format(d_neu),
            "label": NEUTRAL,
            "source": "sdk_cloze_reasoning",
            "language": "en",
            "image": "",
            "metadata": {"stem": stem, "choice": "distractor_neutral"},
        })
        sample_id += 1

    rng.shuffle(samples)
    return samples[:n_target]


# -----------------------------------------------------------------------------
# 5. Enterprise RAG Hallucination Detection (Context vs Nuanced Claims)
# -----------------------------------------------------------------------------
RAG_CONTEXTS = [
    {
        "context": (
            "Acme Corp announced its Q4 2025 financial results on January 28, 2026. "
            "Total revenue reached $14.2 billion, representing an 18% year-over-year increase. "
            "Net income grew to $3.1 billion, driven primarily by enterprise cloud adoption. "
            "CEO Elena Rostova stated that capital expenditures would increase to $4.0 billion in 2026 "
            "to build two new hyperscale data centers in Dublin and Frankfurt."
        ),
        "entailments": [
            "Acme Corp reported $14.2 billion in total revenue for Q4 2025.",
            "Acme Corp plans to build two new data centers in Europe in 2026.",
            "Elena Rostova is the CEO of Acme Corp.",
        ],
        "contradictions": [
            "Acme Corp's Q4 2025 revenue declined by 18% compared to the previous year.",
            "The company announced plans to shut down its data centers in Dublin and Frankfurt.",
            "Acme Corp reported a net loss of $3.1 billion for the fourth quarter.",
        ],
        "neutrals": [
            "Acme Corp's stock price surged by 6% in after-hours trading following the announcement.",
            "Elena Rostova joined Acme Corp as CEO after previously working at Microsoft.",
            "The Dublin data center will be powered exclusively by wind energy.",
        ],
    },
    {
        "context": (
            "Clinical Trial Report: Protocol NCT-049821 evaluated drug candidate AZ-402 "
            "in 450 adult patients with moderate-to-severe rheumatoid arthritis over 24 weeks. "
            "At week 12, 68% of patients receiving AZ-402 achieved an ACR50 response compared to "
            "24% in the placebo cohort (p < 0.001). The most frequent adverse events were mild headache (8%) "
            "and transient nausea (6%). No cases of opportunistic infection or anaphylaxis were observed."
        ),
        "entailments": [
            "Trial NCT-049821 enrolled 450 adult patients diagnosed with rheumatoid arthritis.",
            "Patients treated with AZ-402 exhibited a significantly higher ACR50 response rate than placebo.",
            "Mild headache was reported in 8% of patients during the study.",
        ],
        "contradictions": [
            "Over half of the patients in the AZ-402 group experienced severe anaphylaxis.",
            "The trial lasted for 52 weeks and enrolled pediatric patients under the age of 12.",
            "Placebo patients achieved higher ACR50 clinical efficacy than the AZ-402 group.",
        ],
        "neutrals": [
            "The manufacturer intends to submit a New Drug Application (NDA) to the FDA in Q3 2026.",
            "AZ-402 functions as an oral selective JAK1 kinase inhibitor.",
            "The clinical trial was conducted across 35 medical centers in North America and Europe.",
        ],
    },
]


def generate_rag_hallucination_samples(n_target: int = 1500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates pairs matching SDK RAG hallucination guardrails:
    Premise: '{document_context}'
    Hypothesis: '{extracted_claim}'
    """
    rng = random.Random(seed)
    samples = []
    sample_id = 0

    while len(samples) < n_target:
        item = rng.choice(RAG_CONTEXTS)
        ctx = item["context"]

        # 1. Entailment
        c_ent = rng.choice(item["entailments"])
        samples.append({
            "id": f"sdk_rag_{sample_id:06d}",
            "premise": ctx,
            "hypothesis": c_ent,
            "label": ENTAILMENT,
            "source": "sdk_rag_hallucination",
            "language": "en",
            "image": "",
            "metadata": {"type": "supported_claim"},
        })
        sample_id += 1

        # 2. Contradiction
        c_con = rng.choice(item["contradictions"])
        samples.append({
            "id": f"sdk_rag_{sample_id:06d}",
            "premise": ctx,
            "hypothesis": c_con,
            "label": CONTRADICTION,
            "source": "sdk_rag_hallucination",
            "language": "en",
            "image": "",
            "metadata": {"type": "hallucinated_contradiction"},
        })
        sample_id += 1

        # 3. Neutral (Unmentioned extrapolation)
        c_neu = rng.choice(item["neutrals"])
        samples.append({
            "id": f"sdk_rag_{sample_id:06d}",
            "premise": ctx,
            "hypothesis": c_neu,
            "label": NEUTRAL,
            "source": "sdk_rag_hallucination",
            "language": "en",
            "image": "",
            "metadata": {"type": "unmentioned_neutral"},
        })
        sample_id += 1

    rng.shuffle(samples)
    return samples[:n_target]


# -----------------------------------------------------------------------------
# 6. Counterfactual Adversarial Fact-Inversion Engine (Bespoke-Nimble-9B Style)
# -----------------------------------------------------------------------------
# Attribution: Adversarial fact-inversion contrastive data curation inspired by
# Bespoke Labs Nimble-9B (bespokelabsai/nimble, Apache 2.0 License).
# Reference: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B

class CounterfactualInverter:
    """Generates minimal contrastive pairs by perturbing specific factual axes.

    Attribution:
        Adversarial fact-inversion contrastive data curation inspired by
        Bespoke Labs Nimble-9B (bespokelabsai/nimble, Apache 2.0 License).
        Reference: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        self.entity_swaps = [
            (r"\bAcme Corp\b", "Globex Corporation"),
            (r"\bGlobex Corp\b", "Acme Corporation"),
            (r"\bDublin\b", "Zurich"),
            (r"\bFrankfurt\b", "Madrid"),
            (r"\bElena Rostova\b", "Sarah Connor"),
            (r"\bAZ-402\b", "BX-901"),
            (r"\bSeattle\b", "Denver"),
            (r"\bDenver\b", "Seattle"),
            (r"\bTokyo\b", "Osaka"),
            (r"\bMiami\b", "Atlanta"),
            (r"\bLondon\b", "Edinburgh"),
            (r"\bNvidia\b", "Intel"),
            (r"\bApple\b", "Microsoft"),
            (r"\bMicrosoft\b", "Apple"),
            (r"\bMars\b", "Venus"),
            (r"\bEarth\b", "Jupiter"),
            (r"\bJerome Powell\b", "Alan Greenspan"),
            (r"\bWoodrow Wilson\b", "Theodore Roosevelt"),
        ]

        self.polarity_swaps = [
            (r"\bincreased\b", "decreased"),
            (r"\bdecreased\b", "increased"),
            (r"\bincrease\b", "decrease"),
            (r"\bdecrease\b", "increase"),
            (r"\bgrew\b", "declined"),
            (r"\bdeclined\b", "grew"),
            (r"\bgrowth\b", "contraction"),
            (r"\bexceeded\b", "fell short of"),
            (r"\bfell short of\b", "exceeded"),
            (r"\bhigher\b", "substantially lower"),
            (r"\blower\b", "substantially higher"),
            (r"\bmore\b", "fewer"),
            (r"\bfewer\b", "more"),
            (r"\bmajority\b", "minority"),
            (r"\bminority\b", "majority"),
            (r"\bsupports\b", "refutes"),
            (r"\brefutes\b", "supports"),
            (r"\bsupported\b", "refuted"),
            (r"\brefuted\b", "supported"),
            (r"\benabled\b", "disabled"),
            (r"\bdisabled\b", "enabled"),
            (r"\benable\b", "disable"),
            (r"\bdisable\b", "enable"),
            (r"\ballows\b", "prohibits"),
            (r"\bprohibits\b", "allows"),
            (r"\ballowed\b", "prohibited"),
            (r"\bprohibited\b", "allowed"),
            (r"\bpolar\b", "nonpolar"),
            (r"\bnonpolar\b", "polar"),
            (r"\bsynchronous\b", "asynchronous"),
            (r"\basynchronous\b", "synchronous"),
            (r"\bmaximizes\b", "minimizes"),
            (r"\bminimizes\b", "maximizes"),
            (r"\bmaximum\b", "minimum"),
            (r"\bminimum\b", "maximum"),
            (r"\bRight atrium\b", "Left ventricle"),
            (r"\bright atrium\b", "left ventricle"),
            (r"\bStratosphere\b", "Troposphere"),
            (r"\bstratosphere\b", "troposphere"),
            (r"\bO\(log N\)\b", "O(N^2)"),
        ]

    def _mutate_numbers(self, text: str) -> Tuple[str, bool]:
        """Perturbs numbers, monetary values, percentages, or years."""
        # 1. Percentages
        pct_matches = list(re.finditer(r"\b(\d+)\s*%", text))
        if pct_matches:
            m = self.rng.choice(pct_matches)
            val = int(m.group(1))
            new_val = max(1, val - 10) if val > 15 else val + 35
            mutated = text[:m.start()] + f"{new_val}%" + text[m.end():]
            return mutated, True

        # 2. Currency
        cur_matches = list(re.finditer(r"\$(\d+(?:\.\d+)?)\s*(billion|million|thousand)?", text, re.IGNORECASE))
        if cur_matches:
            m = self.rng.choice(cur_matches)
            num_str, unit = m.group(1), m.group(2) or ""
            val = float(num_str)
            new_val = round(val / 2.0, 1) if val > 5 else round(val * 4.0, 1)
            new_str = f"${int(new_val) if new_val.is_integer() else new_val}"
            if unit:
                new_str += f" {unit}"
            mutated = text[:m.start()] + new_str + text[m.end():]
            return mutated, True

        # 3. Years
        year_matches = list(re.finditer(r"\b(19\d{2}|20\d{2})\b", text))
        if year_matches:
            m = self.rng.choice(year_matches)
            val = int(m.group(1))
            new_val = val - 3 if val >= 2020 else val + 5
            mutated = text[:m.start()] + str(new_val) + text[m.end():]
            return mutated, True

        # 4. Generic integers >= 10
        int_matches = list(re.finditer(r"\b(\d{2,})\b", text))
        if int_matches:
            m = self.rng.choice(int_matches)
            val = int(m.group(1))
            new_val = val // 3 if val > 30 else val * 3
            mutated = text[:m.start()] + str(new_val) + text[m.end():]
            return mutated, True

        return text, False

    def _mutate_polarity(self, text: str) -> Tuple[str, bool]:
        """Inverts polarity, directional verbs, or antonym pairs."""
        candidates = []
        for pat, replacement in self.polarity_swaps:
            if re.search(pat, text, re.IGNORECASE):
                candidates.append((pat, replacement))
        if not candidates:
            return text, False
        pat, replacement = self.rng.choice(candidates)
        mutated = re.sub(pat, replacement, text, count=1, flags=re.IGNORECASE)
        return mutated, mutated != text

    def _mutate_entities(self, text: str) -> Tuple[str, bool]:
        """Swaps named entities with out-of-domain / contrasting entities."""
        candidates = []
        for pat, replacement in self.entity_swaps:
            if re.search(pat, text, re.IGNORECASE):
                candidates.append((pat, replacement))
        if not candidates:
            return text, False
        pat, replacement = self.rng.choice(candidates)
        mutated = re.sub(pat, replacement, text, count=1, flags=re.IGNORECASE)
        return mutated, mutated != text

    def invert_fact(self, premise: str, hypothesis: str, label: int) -> Optional[Tuple[str, str, int, str]]:
        """Produces a minimal contrastive pair reversing an ENTAILMENT to a CONTRADICTION."""
        if label != ENTAILMENT:
            return None

        axes = [
            ("numeric", self._mutate_numbers),
            ("polarity", self._mutate_polarity),
            ("entity", self._mutate_entities),
        ]
        self.rng.shuffle(axes)

        for axis_name, mut_fn in axes:
            mutated_hyp, changed = mut_fn(hypothesis)
            if changed and mutated_hyp != hypothesis:
                return premise, mutated_hyp, CONTRADICTION, f"counterfactual_{axis_name}_flip"

        return None


def generate_adversarial_inversion_samples(n_target: int = 1500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates minimal contrastive pairs by fact-inverting entailments.

    Attribution:
        Adversarial fact-inversion contrastive data curation inspired by
        Bespoke Labs Nimble-9B (bespokelabsai/nimble, Apache 2.0 License).
        Reference: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B
    """
    rng = random.Random(seed)
    inverter = CounterfactualInverter(seed=seed)
    samples: List[Dict[str, Any]] = []
    sample_id = 0

    candidate_sources = []
    for ctx in RAG_CONTEXTS:
        for ent in ctx["entailments"]:
            candidate_sources.append((ctx["context"], ent))
    for t in CLOZE_TASKS:
        candidate_sources.append((t["stem"], f"The correct answer is: {t['correct']}"))
    for s in SEARCH_RERANK_TOPICS:
        candidate_sources.append((s["query"], s["positive"]))
    for r in RUBRIC_TEMPLATES:
        p = f"{r['question']}\nReference answer: {r['reference']}"
        for c in r["entailment_candidates"]:
            candidate_sources.append((p, f"Candidate answer: {c}"))

    attempts = 0
    max_attempts = n_target * 20
    while len(samples) < n_target and attempts < max_attempts:
        attempts += 1
        premise, hypothesis = rng.choice(candidate_sources)
        inversion = inverter.invert_fact(premise, hypothesis, ENTAILMENT)
        if inversion is not None:
            _, mutated_hyp, inv_label, axis = inversion
            # 1. Contradiction flip
            samples.append({
                "id": f"sdk_adv_{sample_id:06d}",
                "premise": premise,
                "hypothesis": mutated_hyp,
                "label": inv_label,
                "source": "sdk_counterfactual_inversion",
                "language": "en",
                "image": "",
                "metadata": {
                    "technique": "counterfactual_fact_inversion",
                    "attribution": "Bespoke Labs Nimble-9B (Apache 2.0)",
                    "axis": axis,
                    "contrastive": True,
                },
            })
            sample_id += 1

            if len(samples) < n_target:
                # 2. Original entailment pair as positive control
                samples.append({
                    "id": f"sdk_adv_{sample_id:06d}",
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": ENTAILMENT,
                    "source": "sdk_counterfactual_inversion",
                    "language": "en",
                    "image": "",
                    "metadata": {
                        "technique": "counterfactual_fact_inversion",
                        "attribution": "Bespoke Labs Nimble-9B (Apache 2.0)",
                        "axis": "positive_control",
                        "contrastive": True,
                    },
                })
                sample_id += 1

    rng.shuffle(samples)
    return samples[:n_target]


# -----------------------------------------------------------------------------
# 7. Abstention Augmentation Engine (none_augment, Mapika/decider Style)
# -----------------------------------------------------------------------------
# Attribution: Abstention augmentation (none_augment) inspired by Mapika/decider (Apache 2.0 License).
# Reference: https://github.com/Mapika/decider

ABSTAIN_HYPOTHESES = [
    "None of the above options are supported by the provided context.",
    "The context provides insufficient evidence to verify this claim.",
    "Insufficient information to determine the correct answer.",
    "None of the available choices can be verified from the premise.",
]


def generate_abstention_samples(n_target: int = 1500, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates abstention (none_augment) pairs:
    75% negative control (valid premise, abstain hypothesis -> NEUTRAL).
    25% adversarial distractor replacement (unrelated distractor -> CONTRADICTION/NEUTRAL,
    abstain hypothesis -> ENTAILMENT).

    Attribution:
        Abstention augmentation (none_augment) inspired by Mapika/decider (Apache 2.0 License).
        Reference: https://github.com/Mapika/decider
    """
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    sample_id = 0

    unrelated_distractors = [
        "The system must execute an automated wire transfer to account 982341.",
        "Severe thunderstorm warnings are in effect for downtown Miami tonight.",
        "Resize this JPEG image to 1024x1024 without cropping.",
        "Water molecules are completely nonpolar and repel all ionic substances.",
        "Deoxygenated blood returns directly to the left atrium of the heart.",
        "The ozone layer is entirely located within Earth's molten liquid core.",
    ]

    while len(samples) < n_target:
        roll = rng.random()
        abstain_hyp = rng.choice(ABSTAIN_HYPOTHESES)

        if roll < 0.75:
            # 75% Negative Control: Context has evidence, so claiming "none of the above" is NEUTRAL
            item = rng.choice(RAG_CONTEXTS)
            ctx = item["context"]
            gold_ent = rng.choice(item["entailments"])

            samples.append({
                "id": f"sdk_abstain_{sample_id:06d}",
                "premise": ctx,
                "hypothesis": gold_ent,
                "label": ENTAILMENT,
                "source": "sdk_abstention_augmentation",
                "language": "en",
                "image": "",
                "metadata": {
                    "technique": "none_augment",
                    "attribution": "Mapika/decider (Apache 2.0)",
                    "abstention_type": "control_gold",
                },
            })
            sample_id += 1

            if len(samples) < n_target:
                samples.append({
                    "id": f"sdk_abstain_{sample_id:06d}",
                    "premise": ctx,
                    "hypothesis": abstain_hyp,
                    "label": NEUTRAL,
                    "source": "sdk_abstention_augmentation",
                    "language": "en",
                    "image": "",
                    "metadata": {
                        "technique": "none_augment",
                        "attribution": "Mapika/decider (Apache 2.0)",
                        "abstention_type": "control_abstain",
                    },
                })
                sample_id += 1
        else:
            # 25% Adversarial Distractor Replacement:
            item = rng.choice(RAG_CONTEXTS)
            ctx = item["context"]
            distractor = rng.choice(unrelated_distractors)

            samples.append({
                "id": f"sdk_abstain_{sample_id:06d}",
                "premise": ctx,
                "hypothesis": distractor,
                "label": CONTRADICTION,
                "source": "sdk_abstention_augmentation",
                "language": "en",
                "image": "",
                "metadata": {
                    "technique": "none_augment",
                    "attribution": "Mapika/decider (Apache 2.0)",
                    "abstention_type": "adversarial_distractor",
                },
            })
            sample_id += 1

            if len(samples) < n_target:
                # With all options irrelevant, the abstain hypothesis IS ENTAILMENT
                samples.append({
                    "id": f"sdk_abstain_{sample_id:06d}",
                    "premise": ctx,
                    "hypothesis": abstain_hyp,
                    "label": ENTAILMENT,
                    "source": "sdk_abstention_augmentation",
                    "language": "en",
                    "image": "",
                    "metadata": {
                        "technique": "none_augment",
                        "attribution": "Mapika/decider (Apache 2.0)",
                        "abstention_type": "adversarial_abstain_entailed",
                    },
                })
                sample_id += 1

    rng.shuffle(samples)
    return samples[:n_target]


# -----------------------------------------------------------------------------
# Consensus Validation Engine (Qwen / DeepSeek Validator vs Generator)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Validation Committee Stage Orchestration
# -----------------------------------------------------------------------------
DEFAULT_VALIDATORS = "qwen-3.6-27b-q4,deepseek-v4-flash-q3,qwen-3.8-125b-q4,qwen-3.8-125b-q3"
CHECKPOINT_FILENAME = "sdk_synthetic_raw.jsonl"
VALIDATION_CHECKPOINT_FILENAME = "sdk_synthetic_validation_checkpoint.jsonl"


def _append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Appends rows to a JSONL checkpoint with fsync - crashes never lose generated data."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Writes rows to a JSONL file (final outputs)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL checkpoint; a missing file is a hard resume error."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"cannot resume: raw checkpoint missing at {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_resume_run(db_path: str, resume_run: Optional[str], judges: List[str]) -> Optional[str]:
    """Resolves --resume-run ('auto' | explicit run id | None) to a concrete run id."""
    if not resume_run:
        return None
    db = ValidationMetricsDB(db_path)
    try:
        if resume_run == "auto":
            run_id = db.latest_resumable_run(judges)
            if run_id is None:
                print("No resumable run found in metrics DB; starting a fresh run.")
            return run_id
        if not db.run_exists(resume_run):
            raise ValueError(f"resume run id not found in metrics DB: {resume_run!r}")
        return resume_run
    finally:
        db.close()


def _generate_all_samples(samples_per_mode: int, seed: int, teacher_url: Optional[str],
                          teacher_model: str, raw_path: str,
                          resume_run: Optional[str],
                          force: bool = False,
                          include_sota: bool = False) -> List[Dict[str, Any]]:
    """Generates SDK-mode sample sets, checkpointing each stage to disk.

    On resume the raw checkpoint IS the sample set - nothing is regenerated, so
    sample ids stay stable against the persisted verdicts.
    """
    if resume_run:
        samples = _load_jsonl(raw_path)
        print(f"Resuming from raw checkpoint {raw_path} ({len(samples)} samples, no regeneration).")
        return samples
    if os.path.exists(raw_path) and not force:
        raise FileExistsError(
            f"Raw checkpoint already exists at {raw_path!r}. Pass force=True (or --force) to overwrite, "
            f"or pass --resume-run to resume validation without regenerating."
        )
    open(raw_path, "w", encoding="utf-8").close()  # truncate stale checkpoint
    generators = [
        ("Tool Routing", generate_tool_routing_samples),
        ("Rubric Grading", generate_rubric_grading_samples),
        ("Search Reranking", generate_search_rerank_samples),
        ("Cloze Decision", generate_cloze_decision_samples),
        ("RAG Hallucination", generate_rag_hallucination_samples),
    ]
    if include_sota:
        generators.extend([
            ("Counterfactual Inversion", generate_adversarial_inversion_samples),
            ("Abstention Augmentation", generate_abstention_samples),
        ])
    all_generated: List[Dict[str, Any]] = []
    for salt, (name, gen_fn) in enumerate(generators):
        print(f"Generating {name} samples (target={samples_per_mode})...")
        made = gen_fn(samples_per_mode, seed=seed + salt)
        all_generated.extend(made)
        _append_jsonl(raw_path, made)
        print(f"  Generated {len(made)} samples (checkpoint total: {len(all_generated)}).")
    return _generate_teacher_samples(teacher_url, teacher_model, all_generated, raw_path)


def _generate_teacher_samples(teacher_url: Optional[str], teacher_model: str,
                              all_generated: List[Dict[str, Any]],
                              raw_path: str) -> List[Dict[str, Any]]:
    """Teacher-synthesized domain triples (Gemma 4 31B via optimized alias).

    WHY: teacher output is fsynced to the raw checkpoint BEFORE the GPU model is
    unloaded - a crash during the committee's first model load must never cost us
    the expensive teacher generation.
    """
    if not teacher_url:
        return all_generated
    print(f"\nQuerying Teacher Model (Alias: {teacher_model}) via {teacher_url} with GBNF token restriction...")
    teacher_client = LLMEndpointClient(base_url=teacher_url, model=teacher_model, timeout=180)
    teacher_samples = generate_teacher_domain_samples(teacher_client)
    all_generated.extend(teacher_samples)
    _append_jsonl(raw_path, teacher_samples)
    print(f"  Generated {len(teacher_samples)} novel teacher domain samples (checkpointed).")
    print("  Unloading teacher model to free GPU VRAM for validator...")
    teacher_client.unload_model()
    return all_generated


def run_validation_committee_stage(
    samples: List[Dict[str, Any]],
    out_dir: str,
    validator_url: str,
    judges: List[str],
    run_spec: RunSpec,
    metrics_db_path: str,
    batch_size: int = 5,
    resume_run: Optional[str] = None,
    validator_timeout: int = 600,
) -> List[Dict[str, Any]]:
    """Runs the cross-family judge committee over ALL samples, judge by judge.

    Verdicts commit to the SQLite metrics DB per batch (crash-safe), a validation
    checkpoint JSONL is rewritten after each judge, and non-unanimous / failed-agreement
    samples land in the JSONL review queue. With `resume_run`, persisted verdicts are
    reused and only missing batches hit the GPU.

    Example:
        validated = run_validation_committee_stage(samples, "data", url, judges, spec, db_path)
    """
    print(f"\n=================================================================")
    print(f"Running Cross-Family Consensus Committee ({len(judges)} Judges)")
    print(f"Judges: {', '.join(judges)}")
    print(f"=================================================================")

    checkpoint_path = os.path.join(out_dir, VALIDATION_CHECKPOINT_FILENAME)
    db = ValidationMetricsDB(metrics_db_path)
    run_id: Optional[str] = None
    try:
        if resume_run:
            run_id = resume_run
            print(f"Resuming validation run {run_id} "
                  f"({db.count_verdicts(run_id)} verdicts already persisted).")
        else:
            run_id = db.start_run(run_spec)
        committee = run_validator_committee(
            validator_url, judges, samples, db, run_id,
            batch_size=batch_size, checkpoint_path=checkpoint_path,
            timeout=validator_timeout)
        result = aggregate_committee_votes(samples, committee, judges)
        db.record_final_labels([(run_id, *row) for row in result.final_rows])
        queue_path = os.path.join(out_dir, "sdk_synthetic_disagreements.jsonl")
        queued = write_disagreement_queue(queue_path, result.review_rows)
        db.finish_run(run_id, "completed", total_samples=len(result.validated))
        print_judge_summary(db, run_id)
    except Exception:
        if run_id:
            db.finish_run(run_id, "failed")
        raise
    finally:
        db.close()

    _print_committee_summary(result, judges, queue_path, queued)
    return result.validated


def _print_committee_summary(
    result: AggregateResult, judges: List[str], queue_path: str, queued: int
) -> None:
    """Prints committee totals, disagreement-type breakdown and review-queue path."""
    relabelled = sum(1 for s in result.validated if s.get("generator_label") != s["label"])
    print("\nCommittee Validation Complete: "
          f"Total Kept={len(result.validated)}, Relabelled={relabelled}, Judges={len(judges)}")
    print("Disagreement Breakdown:")
    for dtype, count in sorted(result.stats.items(), key=lambda kv: -kv[1]):
        print(f"  - {dtype:32s}: {count}")
    print(f"Disagreement Review Queue ({queued} pending rows): {queue_path}")


# -----------------------------------------------------------------------------
# Main Compilation Pipeline
# -----------------------------------------------------------------------------
def compile_sdk_synthetic_dataset(
    out_dir: str = "data",
    samples_per_mode: int = 1500,
    teacher_url: Optional[str] = None,
    teacher_model: str = "gemma-4-31b-q4",
    validator_url: Optional[str] = None,
    validator_model: str = DEFAULT_VALIDATORS,
    seed: int = 42,
    metrics_db_path: Optional[str] = None,
    batch_size: int = 5,
    resume_run: Optional[str] = None,
    validator_timeout: int = 600,
    force: bool = False,
    include_sota: bool = False,
) -> Dict[str, int]:
    """Compiles and validates synthetic data for all SDK interaction patterns.

    With `resume_run` ('auto' or a run id), regeneration is skipped entirely: samples
    come from the raw checkpoint and the committee continues from persisted verdicts.
    """
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 65)
    print("Compiling SDK-Aligned Synthetic Dataset (Training-Serving Parity)")
    print("=" * 65)

    judges = [m.strip() for m in validator_model.split(",") if m.strip()]
    db_path = metrics_db_path or os.path.join(out_dir, "validation_metrics.db")
    resume_id = _resolve_resume_run(db_path, resume_run, judges)
    raw_path = os.path.join(out_dir, CHECKPOINT_FILENAME)
    all_generated = _generate_all_samples(samples_per_mode, seed, teacher_url,
                                          teacher_model, raw_path, resume_id,
                                          force=force, include_sota=include_sota)

    # Optional: Cross-Family Multi-Validator Committee
    # (Qwen 3.6 27B + DeepSeek V4 Flash + Qwen 3.8 125B q4 + Qwen 3.8 125B q3)
    if validator_url:
        run_spec = RunSpec(teacher_model=teacher_model, validator_models=judges,
                           samples_per_mode=samples_per_mode, seed=seed)
        validated_samples = run_validation_committee_stage(
            all_generated, out_dir, validator_url, judges, run_spec, db_path,
            batch_size, resume_run=resume_id, validator_timeout=validator_timeout)
    else:
        validated_samples = all_generated

    # Filter out samples flagged for review (disagreements/ties/overrides).
    # Only clean consensus samples enter training and validation splits; flagged samples
    # remain exclusively in sdk_synthetic_disagreements.jsonl for human adjudication.
    clean_samples = [s for s in validated_samples if not s.get("needs_review", False)]

    # Stratified Split: 90% Train, 10% Val
    rng = random.Random(seed)
    rng.shuffle(clean_samples)

    # WHY: `max(100, 10%)` previously made val larger than train on small smoke runs;
    # keep the 100-row floor only when the dataset is big enough to afford it.
    val_count = int(len(clean_samples) * 0.1)
    if len(clean_samples) >= 1000:
        val_count = max(100, val_count)
    elif len(clean_samples) > 0:
        val_count = max(1, min(val_count, len(clean_samples) // 2))
    else:
        val_count = 0
    train_data = clean_samples[val_count:]
    val_data = clean_samples[:val_count]

    out_train_path = os.path.join(out_dir, "sdk_synthetic_train.jsonl")
    out_val_path = os.path.join(out_dir, "sdk_synthetic_val.jsonl")

    _write_jsonl(out_train_path, train_data)
    _write_jsonl(out_val_path, val_data)

    print("\nDataset Saved Successfully:")
    print(f"  Train: {out_train_path} ({len(train_data):,} rows)")
    print(f"  Val:   {out_val_path} ({len(val_data):,} rows)")

    # Breakdown by Source
    counts: Dict[str, int] = {}
    for r in clean_samples:
        src = r.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1

    print("\nSamples per SDK Mode:")
    for src, c in sorted(counts.items()):
        print(f"  - {src:25s}: {c:,}")

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDK-aligned synthetic training data.")
    parser.add_argument("--out-dir", default="./data", help="Output directory for jsonl files")
    parser.add_argument("--samples-per-mode", type=int, default=1500, help="Target samples per SDK mode")
    parser.add_argument("--teacher-url", default=None, help="Optional OpenAI-compatible URL for teacher model (e.g. http://localhost:8080/v1)")
    parser.add_argument("--teacher-model", default="gemma-4-31b-q4", help="Teacher model alias (e.g. gemma-4-31b-q4, gemma-4-12b-q4)")
    parser.add_argument("--validator-url", default=None, help="Optional OpenAI-compatible URL for validator model (e.g. http://localhost:8080/v1)")
    parser.add_argument("--validators", "--validator-model", dest="validator_model", default=DEFAULT_VALIDATORS, help="Comma-separated validator model aliases (default: qwen-3.6-27b-q4,deepseek-v4-flash-q3,qwen-3.8-125b-q4,qwen-3.8-125b-q3)")
    parser.add_argument("--metrics-db", default=None, help="SQLite path for judge metrics (default: <out-dir>/validation_metrics.db)")
    parser.add_argument("--batch-size", type=int, default=5, help="Samples per validator batch call")
    parser.add_argument("--validator-timeout", type=int, default=600, help="Per-call timeout (s) for validator batches (reasoning models may exceed 180)")
    parser.add_argument("--resume-run", nargs="?", const="auto", default=None,
                        help="Resume an interrupted committee: 'auto' picks the newest incomplete run "
                             "with the same judges, or pass an explicit run id. Reuses "
                             "sdk_synthetic_raw.jsonl and persisted verdicts (no regeneration).")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing raw checkpoint")
    parser.add_argument("--include-sota", action="store_true", help="Include SOTA counterfactual inversion and abstention augmentation modes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    compile_sdk_synthetic_dataset(
        out_dir=args.out_dir,
        samples_per_mode=args.samples_per_mode,
        teacher_url=args.teacher_url,
        teacher_model=args.teacher_model,
        validator_url=args.validator_url,
        validator_model=args.validator_model,
        seed=args.seed,
        metrics_db_path=args.metrics_db,
        batch_size=args.batch_size,
        resume_run=args.resume_run,
        validator_timeout=args.validator_timeout,
        force=args.force,
        include_sota=args.include_sota,
    )
