from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import grpc
import numpy as np

from ...protos.sid import sid_pb2, sid_pb2_grpc


AudioLike = Union[np.ndarray, bytes, bytearray, memoryview]


def _to_pcm16le_bytes(audio: AudioLike) -> bytes:
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be np.ndarray or bytes-like, got {type(audio)!r}")

    if audio.ndim != 1:
        raise ValueError(f"audio must be mono 1-D array, got shape={audio.shape!r}")

    if np.issubdtype(audio.dtype, np.floating):
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        return pcm.tobytes()

    if audio.dtype == np.int16:
        return audio.tobytes()

    pcm = audio.astype(np.int16)
    return pcm.tobytes()


def _feat_bytes_to_vector(feat_bytes: bytes) -> np.ndarray:
    if len(feat_bytes) == 0:
        raise ValueError("empty feature bytes returned from server")
    if len(feat_bytes) % 4 != 0:
        raise ValueError(
            f"feature bytes length {len(feat_bytes)} is not divisible by 4; "
            "cannot decode as float32 vector"
        )
    return np.frombuffer(feat_bytes, dtype=np.float32).copy()


@dataclass(slots=True)
class Inferencer:
    target: str = "localhost:50017"
    vad_alg: str = ""
    sid_alg: str = ""
    timeout_s: float = 10.0
    max_message_mb: int = 64

    def _channel(self) -> grpc.Channel:
        max_bytes = int(self.max_message_mb) * 1024 * 1024
        return grpc.insecure_channel(
            self.target,
            options=(
                ("grpc.max_send_message_length", max_bytes),
                ("grpc.max_receive_message_length", max_bytes),
            ),
        )

    def feature(self, audio_16k: AudioLike) -> np.ndarray:
        pcm = _to_pcm16le_bytes(audio_16k)

        with self._channel() as channel:
            stub = sid_pb2_grpc.EngineStub(channel)
            req = sid_pb2.FeatRequest(vadAlg=self.vad_alg, sidAlg=self.sid_alg, pcm=pcm)
            resp = stub.Feature(req, timeout=self.timeout_s)

        return _feat_bytes_to_vector(resp.feat)
