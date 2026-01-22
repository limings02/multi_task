from __future__ import annotations

import json
import logging
import os
from typing import Dict

from src.cli.args import build_parser
from src.core.logging import get_logger
from src.data.canonical import build_canonical_aliccp


def _collect_overrides(args) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    mapping = {
        "skeleton_path": "raw.skeleton_path",
        "common_features_path": "raw.common_features_path",
        "out_dir": "canonical.out_dir",
        "sqlite_path": "canonical.sqlite_path",
        "samples_path": "canonical.samples_path",
        "tokens_dir": "canonical.tokens_dir",
        "nrows": "params.nrows",
        "chunksize_sk": "params.chunksize_sk",
        "chunksize_cf": "params.chunksize_cf",
        "buffer_max_tokens": "params.buffer_max_tokens",
    }
    for arg_name, cfg_key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[cfg_key] = value
    if getattr(args, "rebuild_sqlite", False):
        overrides["params.rebuild_sqlite"] = True
    if getattr(args, "overwrite", False):
        overrides["params.overwrite"] = True
    return overrides


def main():
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "log_level", None):
        os.environ["LOGLEVEL"] = args.log_level
        logging.getLogger().setLevel(args.log_level.upper())

    logger = get_logger(__name__)

    if args.command == "canonical":
        overrides = _collect_overrides(args)
        manifest = build_canonical_aliccp(args.config, overrides=overrides)
        manifest_path = (
            os.path.join(manifest["output_dir"], "manifest.json")
            if manifest.get("output_dir")
            else None
        )
        logger.info(
            "Canonical build completed. samples_rows=%s tokens_rows=%s manifest=%s",
            manifest.get("samples_rows"),
            manifest.get("tokens_total_rows"),
            manifest_path,
        )
        print(json.dumps({"manifest": manifest_path, "summary": manifest}, indent=2))
    else:  # pragma: no cover - argparse guards this
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
