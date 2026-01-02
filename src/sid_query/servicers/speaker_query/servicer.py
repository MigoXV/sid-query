from __future__ import annotations

import logging

import grpc

from sid_query.inferencers.speaker_query import AudioInput, QueryConfig, SidFeatureError, SpeakerQueryInferencer
from sid_query.protos.sid_query import sid_query_pb2, sid_query_pb2_grpc

DEFAULT_SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)


def _audio_input_from_proto(
    audio: bytes, audio_format: sid_query_pb2.AudioFormat
) -> AudioInput:
    return AudioInput(
        audio=audio,
        sample_rate=int(audio_format.sample_rate) if audio_format.sample_rate else DEFAULT_SAMPLE_RATE,
        channels=int(audio_format.channels) if audio_format.channels else 1,
        encoding=audio_format.encoding or "pcm16",
    )


def _query_config_from_proto(config: sid_query_pb2.QueryConfig) -> QueryConfig:
    return QueryConfig(
        top_k=int(config.top_k) if config.top_k else 1,
        score_threshold=float(config.score_threshold),
        return_score=bool(config.return_score),
    )


class SpeakerQueryServicer(sid_query_pb2_grpc.SpeakerQueryServiceServicer):
    def __init__(self, inferencer: SpeakerQueryInferencer) -> None:
        self._inferencer = inferencer

    def QuerySpeaker(self, request, context):
        if not request.audio:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "audio is required")

        audio_input = _audio_input_from_proto(request.audio, request.audio_format)
        config = _query_config_from_proto(request.query_config)

        try:
            matches = self._inferencer.query(audio_input, config)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except SidFeatureError as exc:
            logger.error("QuerySpeaker inference failed: %s", exc)
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))

        results = []
        for match in matches:
            entity = sid_query_pb2.EntityInfo(
                entity_id=str(match.entity.entity_id),
                entity_type=match.entity.entity_type,
                metadata=match.entity.metadata,
            )
            results.append(
                sid_query_pb2.MatchEntity(entity=entity, score=match.score)
            )

        logger.info(
            "QuerySpeaker success matches=%s top_k=%s",
            len(results),
            config.top_k,
        )
        return sid_query_pb2.QuerySpeakerResponse(matches=results)
