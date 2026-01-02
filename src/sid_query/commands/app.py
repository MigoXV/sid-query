from __future__ import annotations

import logging
import sys
from concurrent import futures
from pathlib import Path

import grpc
import typer

from sid_query.inferencers import (
    SidInferencer,
    SidQueryStore,
    SpeakerQueryInferencer,
    SpeakerRegisteryInferencer,
)
from sid_query.protos.sid_query import sid_query_pb2_grpc
from sid_query.servicers import SpeakerQueryServicer, SpeakerRegisteryServicer

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)


@app.command()
def serve(
    bind: str = typer.Option("0.0.0.0:50051", envvar="SID_QUERY_BIND"),
    db_path: Path = typer.Option("model-bin/sid_query/sid_query.sqlite", envvar="SID_QUERY_DB_PATH"),
    index_path: Path = typer.Option("model-bin/sid_query/sid_query.faiss", envvar="SID_QUERY_INDEX_PATH"),
    sid_target: str = typer.Option("localhost:51007", envvar="SID_QUERY_SID_TARGET"),
    vad_alg: str = typer.Option("", envvar="SID_QUERY_VAD_ALG"),
    sid_alg: str = typer.Option("", envvar="SID_QUERY_SID_ALG"),
    max_workers: int = typer.Option(10, envvar="SID_QUERY_MAX_WORKERS"),
    max_message_mb: int = typer.Option(64, envvar="SID_QUERY_MAX_MESSAGE_MB"),
) -> None:
    _configure_logging()
    max_bytes = int(max_message_mb) * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=(
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ),
    )

    store = SidQueryStore(db_path=db_path, index_path=index_path)
    sid_inferencer = SidInferencer(
        target=sid_target, vad_alg=vad_alg, sid_alg=sid_alg, max_message_mb=max_message_mb
    )
    registry_inferencer = SpeakerRegisteryInferencer(store, sid_inferencer)
    query_inferencer = SpeakerQueryInferencer(store, sid_inferencer)

    sid_query_pb2_grpc.add_SpeakerRegistryServiceServicer_to_server(
        SpeakerRegisteryServicer(registry_inferencer), server
    )
    sid_query_pb2_grpc.add_SpeakerQueryServiceServicer_to_server(
        SpeakerQueryServicer(query_inferencer), server
    )

    server.add_insecure_port(bind)
    server.start()
    logger.info("sid-query serving on %s", bind)
    server.wait_for_termination()


if __name__ == "__main__":
    app()
