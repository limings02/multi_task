from __future__ import annotations

import argparse


def _add_canonical_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/aliccp.yaml",
        help="Path to dataset config YAML.",
    )
    subparser.add_argument("--skeleton-path", type=str, help="Override raw skeleton CSV path.")
    subparser.add_argument(
        "--common-features-path", type=str, help="Override raw common features CSV path."
    )
    subparser.add_argument("--out-dir", type=str, help="Override canonical output root directory.")
    subparser.add_argument("--sqlite-path", type=str, help="Override SQLite output path.")
    subparser.add_argument("--samples-path", type=str, help="Override samples parquet path.")
    subparser.add_argument("--tokens-dir", type=str, help="Override tokens output directory.")
    subparser.add_argument("--nrows", type=int, help="Limit number of skeleton rows to process.")
    subparser.add_argument(
        "--chunksize-sk",
        type=int,
        help="Chunk size for reading skeleton CSV.",
    )
    subparser.add_argument(
        "--chunksize-cf",
        type=int,
        help="Chunk size for reading common features CSV.",
    )
    subparser.add_argument(
        "--buffer-max-tokens",
        type=int,
        help="Max tokens to hold before flushing a parquet part.",
    )
    subparser.add_argument(
        "--rebuild-sqlite",
        action="store_true",
        help="Force rebuild of the SQLite index even if it already exists.",
    )
    subparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing canonical outputs (samples/tokens parts).",
    )
    subparser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Set log level (INFO/DEBUG/WARN). Overrides LOGLEVEL env.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for aliccp-mtl-ranker utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonical_parser = subparsers.add_parser(
        "canonical", help="Build Ali-CCP canonical samples/tokens."
    )
    _add_canonical_args(canonical_parser)

    return parser
