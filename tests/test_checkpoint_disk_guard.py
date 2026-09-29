"""Tests for checkpoint disk management (guard + rolling prune), added after the
2026-09-29 r3 disk-full crash (15GB e4b shards filled a 3.6T disk).
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from finetune import (
    _apply_resume_state,
    _find_latest_resumable,
    _free_gb,
    _prune_step_checkpoints,
    _save_resume_state,
)


class TestFreeGb(unittest.TestCase):
    def test_free_gb_returns_positive_float_for_existing_path(self):
        """_free_gb('.') must return a sane positive number of GB."""
        value = _free_gb(".")
        self.assertIsInstance(value, float)
        self.assertGreater(value, 0.0)

    def test_free_gb_never_raises_on_bad_path(self):
        """A failing stat must degrade to +inf (never block checkpointing)."""
        value = _free_gb("/nonexistent/path/that/cannot/be/statted")
        self.assertEqual(value, float("inf"))


class TestPruneStepCheckpoints(unittest.TestCase):
    def _make_run_dir(self, names):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for name in names:
            os.makedirs(os.path.join(tmp.name, name), exist_ok=True)
        return tmp.name

    def test_prunes_oldest_beyond_window(self):
        """keep=2 keeps step_500/step_750, deletes step_250."""
        run_dir = self._make_run_dir(["step_250", "step_500", "step_750"])
        deleted = _prune_step_checkpoints(run_dir, keep=2)
        self.assertEqual(deleted, ["step_250"])
        self.assertTrue(os.path.isdir(os.path.join(run_dir, "step_500")))
        self.assertTrue(os.path.isdir(os.path.join(run_dir, "step_750")))
        self.assertFalse(os.path.exists(os.path.join(run_dir, "step_250")))

    def test_never_touches_non_step_dirs(self):
        """best/, epoch_1/, stepwise (no digits) and files are untouched."""
        run_dir = self._make_run_dir(["best", "epoch_1", "step_100", "step_200", "step_300"])
        open(os.path.join(run_dir, "step_notes.txt"), "w").close()
        deleted = _prune_step_checkpoints(run_dir, keep=1)
        self.assertEqual(deleted, ["step_100", "step_200"])
        for kept in ("best", "epoch_1", "step_300", "step_notes.txt"):
            self.assertTrue(os.path.exists(os.path.join(run_dir, kept)), kept)

    def test_keep_zero_disables_pruning(self):
        run_dir = self._make_run_dir(["step_1", "step_2", "step_3"])
        self.assertEqual(_prune_step_checkpoints(run_dir, keep=0), [])
        self.assertEqual(len(os.listdir(run_dir)), 3)

    def test_fewer_than_window_deletes_nothing(self):
        run_dir = self._make_run_dir(["step_250"])
        self.assertEqual(_prune_step_checkpoints(run_dir, keep=2), [])

    def test_missing_dir_is_noop(self):
        self.assertEqual(_prune_step_checkpoints("/nonexistent/run/dir", keep=2), [])


class TestResumeStateRoundtrip(unittest.TestCase):
    """Optimizer/scheduler/RNG/position must survive a save-load cycle exactly."""

    def test_roundtrip_restores_schedule_and_position(self):
        param = torch.nn.Parameter(torch.zeros(4))
        opt = torch.optim.AdamW([{"params": [param], "lr": 1e-3}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        for _ in range(30):
            (param.sum() * 2).backward()
            opt.step()
            sched.step()
            opt.zero_grad()

        with tempfile.TemporaryDirectory() as tmp:
            _save_resume_state(tmp, opt, sched, global_step=30, epoch=1, batches_done_in_epoch=412)
            self.assertTrue(os.path.exists(os.path.join(tmp, "resume.pt")))

            opt2 = torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.zeros(4))], "lr": 1e-3}])
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100)
            gstep, epoch, batches = _apply_resume_state(os.path.join(tmp, "resume.pt"), opt2, sched2)

        self.assertEqual((gstep, epoch, batches), (30, 1, 412))
        self.assertAlmostEqual(sched2.get_last_lr()[0], sched.get_last_lr()[0], places=8)
        self.assertEqual(sched2.last_epoch, sched.last_epoch)
        # Adam moment shapes transfer even though opt2 owns a different Parameter object.
        st1 = opt.state[list(opt.state.keys())[0]]
        st2 = opt2.state[list(opt2.state.keys())[0]]
        self.assertEqual(st1["exp_avg"].shape, st2["exp_avg"].shape)

    def test_atomic_marker_missing_means_incomplete(self):
        """A crashed save leaves resume.pt absent -> _find_latest_resumable skips it."""
        with tempfile.TemporaryDirectory() as tmp:
            for name, complete in (("step_250", True), ("step_500", False), ("step_750", True)):
                d = os.path.join(tmp, name)
                os.makedirs(d)
                if complete:
                    open(os.path.join(d, "resume.pt"), "w").close()
            self.assertEqual(_find_latest_resumable(tmp), os.path.join(tmp, "step_750"))

    def test_find_latest_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_find_latest_resumable(tmp))


if __name__ == "__main__":
    unittest.main()
