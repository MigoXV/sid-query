from __future__ import annotations

import argparse
from pathlib import Path

import grpc
import librosa
import numpy as np

from sid_query.inferencers.sid import Inferencer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def add_gaussian_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    signal_power = float(np.mean(audio**2))
    if signal_power == 0.0:
        return audio.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def load_wav_16k_mono(path: Path) -> np.ndarray:
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    if sr != 16000:
        raise ValueError(f"expected 16k after resample, got sr={sr}")
    return y.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="localhost:51007")
    parser.add_argument("--vad-alg", default="")
    parser.add_argument("--sid-alg", default="")
    parser.add_argument("--snr-db", type=float, default=10.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    wav1 = repo_root / "data-bin/test_sid/speaker1_b_cn_16k.wav"
    wav2 = repo_root / "data-bin/test_sid/speaker2_a_cn_16k.wav"

    audio1 = load_wav_16k_mono(wav1)
    audio2 = load_wav_16k_mono(wav2)
    audio1_noisy = add_gaussian_noise(audio1, snr_db=args.snr_db, seed=0)

    infer = Inferencer(target=args.target, vad_alg=args.vad_alg, sid_alg=args.sid_alg)

    try:
        feat1 = infer.feature(audio1)
        feat2 = infer.feature(audio2)
        feat1_noisy = infer.feature(audio1_noisy)
    except grpc.RpcError as e:
        print(f"gRPC call failed: code={e.code()} details={e.details()}")
        return 2

    sim_1_2 = cosine_similarity(feat1, feat2)
    sim_1_1n = cosine_similarity(feat1, feat1_noisy)

    print(f"target: {args.target}")
    print(f"wav1: {wav1.name} feat_dim={feat1.shape[0]}")
    print(f"wav2: {wav2.name} feat_dim={feat2.shape[0]}")
    print(f"cosine(wav1, wav2) = {sim_1_2:.6f}")
    print(f"cosine(wav1, wav1+noise@{args.snr_db}dB) = {sim_1_1n:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
