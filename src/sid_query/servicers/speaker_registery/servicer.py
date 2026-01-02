from __future__ import annotations

import logging

import grpc

from sid_query.inferencers.speaker_registery import AudioInput, SidFeatureError, SpeakerRegisteryInferencer
from sid_query.protos.sid_query import sid_query_pb2, sid_query_pb2_grpc

DEFAULT_SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)


def _parse_entity_id(raw_id: str) -> int:
    try:
        value = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"invalid entity_id {raw_id!r}: must be integer") from exc
    if value < 0:
        raise ValueError(f"invalid entity_id {raw_id!r}: must be >= 0")
    return value


def _audio_input_from_proto(
    audio: bytes, audio_format: sid_query_pb2.AudioFormat
) -> AudioInput:
    return AudioInput(
        audio=audio,
        sample_rate=int(audio_format.sample_rate) if audio_format.sample_rate else DEFAULT_SAMPLE_RATE,
        channels=int(audio_format.channels) if audio_format.channels else 1,
        encoding=audio_format.encoding or "pcm16",
    )


class SpeakerRegisteryServicer(sid_query_pb2_grpc.SpeakerRegistryServiceServicer):
    def __init__(self, inferencer: SpeakerRegisteryInferencer) -> None:
        self._inferencer = inferencer

    def RegisterEntity(self, request, context):
        if not request.audio:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "audio is required")

        try:
            entity_id = _parse_entity_id(request.entity.entity_id) if request.entity.entity_id else None
            audio_input = _audio_input_from_proto(request.audio, request.audio_format)
            metadata = dict(request.entity.metadata)
            new_id = self._inferencer.register(audio_input, metadata, entity_id)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except SidFeatureError as exc:
            logger.error("RegisterEntity inference failed: %s", exc)
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except KeyError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))

        logger.info("RegisterEntity success entity_id=%s", new_id)
        return sid_query_pb2.RegisterEntityResponse(entity_id=str(new_id))

    def UpdateEntity(self, request, context):
        if not request.entity_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "entity_id is required")
        if not request.audio:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "audio is required")

        try:
            entity_id = _parse_entity_id(request.entity_id)
            audio_input = _audio_input_from_proto(request.audio, request.audio_format)
            metadata = dict(request.metadata)
            self._inferencer.update(
                entity_id, audio_input, metadata, overwrite=request.overwrite
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except SidFeatureError as exc:
            logger.error("UpdateEntity inference failed entity_id=%s: %s", request.entity_id, exc)
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except KeyError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))

        logger.info("UpdateEntity success entity_id=%s overwrite=%s", entity_id, request.overwrite)
        return sid_query_pb2.UpdateEntityResponse(entity_id=str(entity_id))

    def DeleteEntity(self, request, context):
        if not request.entity_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "entity_id is required")
        try:
            entity_id = _parse_entity_id(request.entity_id)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        success = self._inferencer.delete(entity_id)
        return sid_query_pb2.DeleteEntityResponse(success=success)

    def GetEntity(self, request, context):
        if not request.entity_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "entity_id is required")
        try:
            entity_id = _parse_entity_id(request.entity_id)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        result = self._inferencer.get(entity_id)
        if not result:
            context.abort(grpc.StatusCode.NOT_FOUND, "entity not found")

        entity = sid_query_pb2.EntityInfo(
            entity_id=str(result.entity_id),
            entity_type=result.entity_type,
            metadata=result.metadata,
        )
        return sid_query_pb2.GetEntityResponse(entity=entity)
