from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Optional

import grpc
import numpy as np
from scipy.io import wavfile

from sid_query.inferencers.sid import Inferencer as SidInferencer
from sid_query.inferencers.speaker_registery.store import SidQueryStore

DEFAULT_SAMPLE_RATE = 16000
ENTITY_TYPE_DEFAULT = "0"


class SidFeatureError(RuntimeError):
    pass


@dataclass(slots=True)
class AudioInput:
    audio: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    encoding: str = "pcm16"


@dataclass(slots=True)
class EntityInfoData:
    entity_id: int
    metadata: Dict[str, str]
    entity_type: str = ENTITY_TYPE_DEFAULT


def _float_from_int_pcm(data: np.ndarray) -> np.ndarray:
    if data.dtype.kind == "f":
        return data.astype(np.float32)
    info = np.iinfo(data.dtype)
    return (data.astype(np.float32) / float(info.max)).clip(-1.0, 1.0)


def _decode_pcm16(audio: bytes, channels: int) -> np.ndarray:
    pcm = np.frombuffer(audio, dtype=np.int16)
    if channels > 1:
        total = (pcm.size // channels) * channels
        pcm = pcm[:total].reshape(-1, channels).mean(axis=1)
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _decode_audio(audio_input: AudioInput) -> np.ndarray:
    if not audio_input.audio:
        raise ValueError("audio is required")
    encoding = (audio_input.encoding or "pcm16").lower()
    sample_rate = int(audio_input.sample_rate) if audio_input.sample_rate else DEFAULT_SAMPLE_RATE
    channels = int(audio_input.channels) if audio_input.channels else 1

    if encoding in ("pcm16", "pcm"):
        audio_f32 = _decode_pcm16(audio_input.audio, channels)
    elif encoding in ("wav", "wav_pcm16"):
        sr, data = wavfile.read(io.BytesIO(audio_input.audio))
        sample_rate = int(sr)
        audio_f32 = _float_from_int_pcm(np.asarray(data))
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.mean(axis=1)
    else:
        raise ValueError(f"unsupported audio encoding: {encoding!r}")

    if sample_rate != DEFAULT_SAMPLE_RATE:
        import librosa

        audio_f32 = librosa.resample(audio_f32, orig_sr=sample_rate, target_sr=DEFAULT_SAMPLE_RATE)

    return np.asarray(audio_f32, dtype=np.float32)


class SpeakerRegisteryInferencer:
    def __init__(self, store: SidQueryStore, sid_inferencer: SidInferencer) -> None:
        self._store = store
        self._sid_inferencer = sid_inferencer

    def _extract_feature(self, audio_input: AudioInput) -> np.ndarray:
        audio = _decode_audio(audio_input)
        try:
            return self._sid_inferencer.feature(audio)
        except (grpc.RpcError, ValueError) as exc:
            raise SidFeatureError(str(exc)) from exc

    def register(self, audio_input: AudioInput, metadata: Dict[str, str], entity_id: Optional[int]) -> int:
        feat = self._extract_feature(audio_input)
        return self._store.register(feat, metadata, entity_id)

    def update(self, entity_id: int, audio_input: AudioInput, metadata: Dict[str, str], overwrite: bool) -> None:
        feat = self._extract_feature(audio_input)
        self._store.update(entity_id, feat, metadata, overwrite)

    def delete(self, entity_id: int) -> bool:
        return self._store.delete(entity_id)

    def get(self, entity_id: int) -> Optional[EntityInfoData]:
        result = self._store.get(entity_id)
        if not result:
            return None
        _, metadata = result
        return EntityInfoData(entity_id=entity_id, metadata=metadata)

