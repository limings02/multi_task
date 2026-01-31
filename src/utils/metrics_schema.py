from __future__ import annotations

from typing import Any, Dict

# Central alias map used for ESMM schema enrichment.
ESMM_ALIASES: Dict[str, str] = {
    "loss_cvr": "loss_ctcvr",
    "auc_cvr": "auc_cvr_click",
    "n_rows": "n_exposure",
    "n_masked_train": "n_click",
    "pos_masked_train": "n_cvr_pos_click",
    "mask_cvr_sum": "n_exposure_batch",
}


def apply_esmm_schema(record: Dict[str, Any], use_esmm: bool) -> Dict[str, Any]:
    """
    Add clearer ESMM field names while keeping legacy flat keys.
    Never mutates the original dict.
    """
    if not use_esmm:
        return record

    enriched = dict(record)

    # Metadata
    enriched.setdefault("schema_version", "metrics_v2_esmm")
    enriched.setdefault("aliases", ESMM_ALIASES)
    enriched.setdefault("loss_targets", ["ctr", "ctcvr"])
    enriched.setdefault("monitor_targets", ["cvr_click"])
    enriched.setdefault("target_def", "esmm: optimize ctr + ctcvr(exposure); monitor cvr(click=1 only)")

    # Copy alias values without removing legacy fields.
    def _copy(dst: str, src: str) -> None:
        if src in record and dst not in enriched:
            enriched[dst] = record[src]

    _copy("loss_ctcvr", "loss_cvr")
    _copy("auc_cvr_click", "auc_cvr")
    _copy("n_exposure", "n_rows")
    _copy("n_exposure_batch", "mask_cvr_sum")
    _copy("n_click", "n_masked_train")
    _copy("n_cvr_pos_click", "pos_masked_train")

    return enriched


__all__ = ["apply_esmm_schema", "ESMM_ALIASES"]
