"""
Tests for the temporal synchronizer.

Generates random joint sequences, shifts them by known offsets,
and verifies the synchronizer recovers the correct alignment.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from synchronize_videos.synchronizer import Synchronizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_random_sequence(T: int, J: int = 22, seed: int = 0) -> torch.Tensor:
    """Generate a smooth random joint trajectory (T x J x 3).

    Uses cumulative sum of small increments so consecutive frames are
    correlated, mimicking real motion.
    """
    gen = torch.Generator().manual_seed(seed)
    increments = torch.randn(T, J, 3, generator=gen) * 0.05
    return increments.cumsum(dim=0)


def shift_sequence(
    seq: torch.Tensor, offset: int, total_len: int
) -> torch.Tensor:
    """Embed *seq* into a longer timeline starting at frame *offset*.

    Returns a window of length total_len that contains the shifted data
    padded with the boundary values (edge-replicate).
    """
    T = seq.shape[0]
    out = torch.zeros(total_len, *seq.shape[1:])
    start = max(offset, 0)
    end = min(offset + T, total_len)
    src_start = start - offset
    src_end = end - offset
    out[start:end] = seq[src_start:src_end]
    # edge-replicate padding
    if start > 0:
        out[:start] = seq[0]
    if end < total_len:
        out[end:] = seq[src_end - 1]
    return out


# ------------------------------------------------------------------ #
#  Pair-wise offset recovery (no noise)
# ------------------------------------------------------------------ #

class TestPairOffsetClean:
    """Recover the offset between two shifted copies of the same sequence."""

    @pytest.fixture
    def sync(self):
        return Synchronizer(device=DEVICE)

    @pytest.mark.parametrize("true_offset", [0, 3, 7, -5, 15])
    def test_recover_pair_offset(self, sync, true_offset):
        J = 22
        T_base = 60
        total_len = T_base + abs(true_offset) + 10

        base = make_random_sequence(T_base, J, seed=42)

        seq1 = shift_sequence(base, 0, total_len).to(DEVICE)
        seq2 = shift_sequence(base, true_offset, total_len).to(DEVICE)

        conf1 = torch.ones(seq1.shape[0], J, device=DEVICE)
        conf2 = torch.ones(seq2.shape[0], J, device=DEVICE)

        estimated = sync.estimate_couple_offset(seq1, seq2, conf1, conf2)
        assert estimated == pytest.approx(true_offset, abs=1), (
            f"Expected offset {true_offset}, got {estimated}"
        )


# ------------------------------------------------------------------ #
#  Multi-camera global alignment (no noise)
# ------------------------------------------------------------------ #

class TestGlobalAlignmentClean:
    """Recover globally consistent start times for K cameras."""

    @pytest.fixture
    def sync(self):
        return Synchronizer(device=DEVICE)

    def test_three_cameras(self, sync):
        J = 22
        T_base = 80
        true_starts = [0, 5, 12]  # ground-truth start frames
        total_len = T_base + max(true_starts) + 5

        base = make_random_sequence(T_base, J, seed=99)

        joints, confs = [], []
        for s in true_starts:
            seq = shift_sequence(base, s, total_len).to(DEVICE)
            joints.append(seq)
            confs.append(torch.ones(seq.shape[0], J, device=DEVICE))

        offset_mat = sync.estimate_offset_matrix(joints, confs)
        estimated = sync.estimate_initial_times(offset_mat)

        # Shift so min is 0 (same convention as the synchronizer)
        true_t = torch.tensor(true_starts, dtype=torch.float32, device=DEVICE)
        true_t = true_t - true_t.min()

        torch.testing.assert_close(
            estimated, true_t, atol=2, rtol=0,
            msg=f"Start times mismatch: estimated {estimated.tolist()} vs true {true_t.tolist()}",
        )

    def test_four_cameras(self, sync):
        J = 17
        T_base = 80
        true_starts = [0, 3, 10, 20]
        total_len = T_base + max(true_starts) + 5

        base = make_random_sequence(T_base, J, seed=77)

        joints, confs = [], []
        for s in true_starts:
            seq = shift_sequence(base, s, total_len).to(DEVICE)
            joints.append(seq)
            confs.append(torch.ones(seq.shape[0], J, device=DEVICE))

        offset_mat = sync.estimate_offset_matrix(joints, confs)
        estimated = sync.estimate_initial_times(offset_mat)

        true_t = torch.tensor(true_starts, dtype=torch.float32, device=DEVICE)
        true_t = true_t - true_t.min()

        torch.testing.assert_close(
            estimated, true_t, atol=2, rtol=0,
            msg=f"Start times mismatch: estimated {estimated.tolist()} vs true {true_t.tolist()}",
        )


# ------------------------------------------------------------------ #
#  Pair-wise offset recovery WITH noise
# ------------------------------------------------------------------ #

class TestPairOffsetNoisy:
    """Same as clean but with additive Gaussian noise on the joints."""

    @pytest.fixture
    def sync(self):
        return Synchronizer(device=DEVICE)

    @pytest.mark.parametrize(
        "true_offset, noise_std",
        [(5, 0.02), (10, 0.05), (-3, 0.03), (0, 0.04)],
    )
    def test_recover_pair_offset_noisy(self, sync, true_offset, noise_std):
        J = 22
        T_base = 80  # longer sequence helps DTW under noise
        total_len = T_base + abs(true_offset) + 10

        base = make_random_sequence(T_base, J, seed=42)

        seq1 = shift_sequence(base, 0, total_len)
        seq2 = shift_sequence(base, true_offset, total_len)

        # Add independent noise to each camera
        torch.manual_seed(100)
        seq1 = seq1 + torch.randn(seq1.shape) * noise_std
        torch.manual_seed(200)
        seq2 = seq2 + torch.randn(seq2.shape) * noise_std

        seq1, seq2 = seq1.to(DEVICE), seq2.to(DEVICE)
        conf1 = torch.ones(seq1.shape[0], J, device=DEVICE)
        conf2 = torch.ones(seq2.shape[0], J, device=DEVICE)

        estimated = sync.estimate_couple_offset(seq1, seq2, conf1, conf2)
        assert estimated == pytest.approx(true_offset, abs=2), (
            f"Expected offset {true_offset}, got {estimated} (noise_std={noise_std})"
        )


# ------------------------------------------------------------------ #
#  Multi-camera global alignment WITH noise
# ------------------------------------------------------------------ #

class TestGlobalAlignmentNoisy:
    """Recover global start times when joints are corrupted by noise."""

    @pytest.fixture
    def sync(self):
        return Synchronizer(device=DEVICE)

    def test_three_cameras_noisy(self, sync):
        J = 22
        T_base = 100
        true_starts = [0, 7, 15]
        noise_std = 0.03
        total_len = T_base + max(true_starts) + 5

        base = make_random_sequence(T_base, J, seed=55)

        joints, confs = [], []
        for idx, s in enumerate(true_starts):
            seq = shift_sequence(base, s, total_len)
            torch.manual_seed(300 + idx)
            seq = seq + torch.randn(seq.shape) * noise_std
            joints.append(seq.to(DEVICE))
            confs.append(torch.ones(seq.shape[0], J, device=DEVICE))

        offset_mat = sync.estimate_offset_matrix(joints, confs)
        estimated = sync.estimate_initial_times(offset_mat)

        true_t = torch.tensor(true_starts, dtype=torch.float32, device=DEVICE)
        true_t = true_t - true_t.min()

        torch.testing.assert_close(
            estimated, true_t, atol=3, rtol=0,
            msg=f"Noisy start times mismatch: estimated {estimated.tolist()} vs true {true_t.tolist()}",
        )

    def test_four_cameras_noisy_with_varying_confidence(self, sync):
        """Some joints have low confidence (simulating occlusion)."""
        J = 22
        T_base = 100
        true_starts = [0, 4, 11, 18]
        noise_std = 0.04
        total_len = T_base + max(true_starts) + 5

        base = make_random_sequence(T_base, J, seed=88)

        joints, confs = [], []
        for idx, s in enumerate(true_starts):
            seq = shift_sequence(base, s, total_len)
            torch.manual_seed(400 + idx)
            seq = seq + torch.randn(seq.shape) * noise_std

            # Random per-joint confidence in [0.3, 1.0]
            gen_conf = torch.Generator().manual_seed(500 + idx)
            conf = 0.3 + 0.7 * torch.rand(seq.shape[0], J, generator=gen_conf)

            joints.append(seq.to(DEVICE))
            confs.append(conf.to(DEVICE))

        offset_mat = sync.estimate_offset_matrix(joints, confs)
        estimated = sync.estimate_initial_times(offset_mat)

        true_t = torch.tensor(true_starts, dtype=torch.float32, device=DEVICE)
        true_t = true_t - true_t.min()

        torch.testing.assert_close(
            estimated, true_t, atol=3, rtol=0,
            msg=f"Noisy+conf start times mismatch: estimated {estimated.tolist()} vs true {true_t.tolist()}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
