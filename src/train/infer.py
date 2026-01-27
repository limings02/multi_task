from __future__ import annotations

"""
Streaming inference helpers that write predictions to Parquet.

The implementation is memory-safe for large validation/test sets by writing
each batch to Parquet incrementally via PyArrow's ParquetWriter.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pyarrow is required for inference export. Install via `pip install pyarrow`."
    ) from exc


def _to_device_features(features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Lightweight variant of train.loops._to_device_features to avoid a hard
    dependency on that private helper.
    """
    fields = {}
    for base, fd in features["fields"].items():
        fields[base] = {
            "indices": fd["indices"].to(device),
            "offsets": fd["offsets"].to(device),
            "weights": fd["weights"].to(device) if fd.get("weights") is not None else None,
        }
    return {"fields": fields, "field_names": features["field_names"]}


def infer_to_parquet(
    model,
    loader,
    device: torch.device,
    out_path: Union[str, Path],
    max_batches: Optional[int] = None,
    include_ctcvr: bool = True,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run inference and stream predictions to Parquet.

    Args:
        model: PyTorch model with "ctr" and "cvr" outputs.
        loader: DataLoader yielding (labels, features, meta).
        device: torch.device to run inference on.
        out_path: output Parquet file path.
        max_batches: optional cap for number of batches (debug/smoke tests).
        include_ctcvr: whether to write pred_ctcvr column.
        split: optional split name for summary metadata.
    """
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    rows_written = 0
    t_start = time.time()

    with torch.no_grad():
        for step, (labels, features, meta) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break

            labels_dev = {k: v.to(device) for k, v in labels.items() if torch.is_tensor(v)}
            features_dev = _to_device_features(features, device)

            outputs = model(features_dev)
            ctr_logit = outputs["ctr"]
            cvr_logit = outputs["cvr"]

            # Normalize shapes to (B,)
            if ctr_logit.dim() > 1:
                ctr_logit = ctr_logit.view(-1)
            if cvr_logit.dim() > 1:
                cvr_logit = cvr_logit.view(-1)

            pred_ctr_prob = torch.sigmoid(ctr_logit).cpu().numpy()
            pred_cvr_prob = torch.sigmoid(cvr_logit).cpu().numpy()

            batch_size = pred_ctr_prob.shape[0]
            data = {
                "entity_id": meta["entity_id"],  # already list[str]
                "row_id": labels_dev["row_id"].cpu().numpy(),
                "y_ctr": labels_dev["y_ctr"].cpu().numpy(),
                "y_cvr": labels_dev["y_cvr"].cpu().numpy(),
                "click_mask": labels_dev["click_mask"].cpu().numpy(),
                "pred_ctr": pred_ctr_prob.astype(np.float32),
                "pred_cvr": pred_cvr_prob.astype(np.float32),
            }
            # y_ctcvr is optional but keep if present for funnel analysis
            if "y_ctcvr" in labels_dev:
                data["y_ctcvr"] = labels_dev["y_ctcvr"].cpu().numpy()

            if include_ctcvr:
                data["pred_ctcvr"] = (pred_ctr_prob * pred_cvr_prob).astype(np.float32)

            table = pa.Table.from_pydict(data)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            rows_written += batch_size

    if writer is not None:
        writer.close()

    return {
        "rows_written": int(rows_written),
        "out_path": str(out_path),
        "split": split,
        "duration_sec": time.time() - t_start,
    }


__all__ = ["infer_to_parquet"]
