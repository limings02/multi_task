from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.core.logging import get_logger
from src.eda.extra.common import write_json, write_table, MULTIHOT_NNZ_THRESHOLD


logger = get_logger(__name__)

MAX_LEN_CAP_HEAVY = 60
MAX_LEN_CAP_LIGHT = 80


def _load_featuremap(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_multihot(field_stats_row: pd.Series, existing_feature: Optional[Dict]) -> bool:
    if existing_feature and existing_feature.get("is_multi_hot"):
        return True
    return float(field_stats_row.get("avg_nnz_per_covered_row", 1.0) or 1.0) > MULTIHOT_NNZ_THRESHOLD


def _token_share(df: pd.DataFrame) -> pd.Series:
    if "token_rows" in df.columns:
        total = df["token_rows"].sum()
        return df["token_rows"] / total if total else df["token_rows"]
    total_cov = (df["coverage_rate"] * df["avg_nnz_per_covered_row"]).sum()
    return (df["coverage_rate"] * df["avg_nnz_per_covered_row"]) / total_cov


def _select_max_len(
    trunc_df: pd.DataFrame,
    quant_df: pd.DataFrame,
    field: str,
    src: int,
    token_share: float,
    fallback: Optional[int],
) -> Optional[int]:
    cap = MAX_LEN_CAP_HEAVY if token_share >= 0.2 else MAX_LEN_CAP_LIGHT
    subset = trunc_df[(trunc_df["field"] == field) & (trunc_df["src"] == src)]
    if subset.empty:
        return min(cap, fallback) if fallback else None
    subset = subset.sort_values("K")
    subset_cap = subset[subset["K"] <= cap]
    good = subset_cap[
        (subset_cap["retained_token_frac"] >= 0.995) & (subset_cap["retained_row_full_frac"] >= 0.95)
    ]
    if not good.empty:
        return int(good.iloc[0]["K"])
    if not subset_cap.empty:
        best = subset_cap.sort_values(["retained_token_frac", "retained_row_full_frac", "K"], ascending=[False, False, True]).iloc[0]
        return int(best["K"])
    p99 = quant_df[(quant_df["field"] == field) & (quant_df["src"] == src)]
    if not p99.empty and pd.notna(p99.iloc[0]["p99"]):
        return int(min(cap, int(math.ceil(p99.iloc[0]["p99"]))))
    return None


def fmt(x, nd: int = 4) -> str:
    try:
        if x is None:
            return "NA"
        if isinstance(x, str):
            return x
        if isinstance(x, (float, int)):
            if math.isnan(x):
                return "NA"
            return f"{x:.{nd}f}"
        val = float(x)
        if math.isnan(val):
            return "NA"
        return f"{val:.{nd}f}"
    except Exception:
        return "NA"


def _select_bucket(bucket_reco: Dict[str, Dict[str, object]], field: str, src: int) -> Optional[int]:
    key = f"{field}|{src}"
    if key not in bucket_reco:
        return None
    return int(bucket_reco[key]["recommended_bucket_local"])


def _select_head_size(oov_summary: pd.DataFrame, field: str, src: int) -> Optional[int]:
    row = oov_summary[(oov_summary["field"] == field) & (oov_summary["src"] == src)]
    if row.empty:
        return None
    val = row.iloc[0]["recommended_head_size"]
    return int(val) if pd.notna(val) else None


def _select_encoding(oov_summary: pd.DataFrame, field: str, src: int, existing: Optional[Dict]) -> str:
    row = oov_summary[(oov_summary["field"] == field) & (oov_summary["src"] == src)]
    if not row.empty and pd.notna(row.iloc[0]["recommended_encoding_hint"]):
        return str(row.iloc[0]["recommended_encoding_hint"])
    if existing and existing.get("encoding"):
        return existing["encoding"]
    return "hash"


def build_featuremap_patch(
    field_stats_df: pd.DataFrame,
    oov_summary_df: pd.DataFrame,
    oov_curve_df: pd.DataFrame,
    quant_df: pd.DataFrame,
    trunc_df: pd.DataFrame,
    hash_collision_df: pd.DataFrame,
    hash_bucket_json: Dict,
    val_fields: List[str],
    psi_df: pd.DataFrame,
    featuremap_path: Path,
) -> Tuple[Dict, str, str]:
    existing_fm = _load_featuremap(featuremap_path)
    existing_defaults = existing_fm.get("defaults", {}) if existing_fm else {
        "dtype": "int64",
        "embedding_dim": 16,
        "pooling": "sum_sqrtn",
        "use_value": False,
        "is_multi_hot": False,
        "max_len": None,
        "vocab": {"pad_token": "__PAD__", "oov_token": "__OOV__"},
        "hash": {"seed": 17},
    }
    existing_feat_map = {(f["field"], f.get("src", 0)): f for f in existing_fm.get("features", [])} if existing_fm else {}

    bucket_reco = {f"{item['field']}|{item['src']}": item for item in hash_bucket_json.get("per_field", [])}

    token_share = _token_share(field_stats_df)
    field_stats_df = field_stats_df.assign(token_share=token_share.values)
    top_token_fields = set(
        zip(
            field_stats_df.sort_values("token_share", ascending=False).head(3)["field"],
            field_stats_df.sort_values("token_share", ascending=False).head(3)["src"],
        )
    )

    features = []
    rationales = []
    for _, row in field_stats_df.iterrows():
        field = row["field"]
        src = int(row["src"])
        existing = existing_feat_map.get((field, src))
        encoding = _select_encoding(oov_summary_df, field, src, existing)
        head_size = _select_head_size(oov_summary_df, field, src)
        hash_bucket = _select_bucket(bucket_reco, field, src)
        is_multi_hot = _is_multihot(row, existing)
        max_len = None
        if is_multi_hot:
            max_len = _select_max_len(
                trunc_df,
                quant_df,
                field,
                src,
                token_share=row.get("token_share", 0.0),
                fallback=existing.get("max_len") if existing else None,
            )
            if max_len is None:
                logger.warning("No truncation curve for multi-hot field=%s src=%s; downgrade to single-hot", field, src)
                is_multi_hot = False
        use_value = existing.get("use_value", False) if existing else (field in val_fields)
        pooling = existing.get("pooling", "sum_sqrtn")
        if is_multi_hot and use_value and "weighted" not in pooling:
            pooling = "weighted_sum_sqrtn"
        emb_dim = existing.get("embedding_dim", existing_defaults.get("embedding_dim", 16))
        if (field, src) in top_token_fields and emb_dim > 8:
            emb_dim = 8
        feature_entry = {
            "field": field,
            "src": src,
            "encoding": encoding,
            "embedding_dim": emb_dim,
            "is_multi_hot": is_multi_hot,
            "max_len": max_len,
            "use_value": use_value,
            "pooling": pooling,
        }
        if encoding == "vocab" and head_size:
            feature_entry["vocab_size"] = head_size
            feature_entry["vocab_head_size"] = head_size
        if encoding == "hybrid":
            feature_entry["hybrid_rule"] = "topn"
            if head_size:
                feature_entry["vocab_head_size"] = head_size
                feature_entry["vocab_size"] = head_size
            feature_entry["tail_encoding"] = "hash"
        if encoding in ("hash", "hybrid") and hash_bucket:
            feature_entry["hash_bucket"] = hash_bucket
        features.append(feature_entry)

        psi_row = psi_df[(psi_df["field"] == field) & (psi_df["src"] == src)]
        psi_val = None if psi_row.empty else psi_row.iloc[0]["psi"]
        oov_row = oov_curve_df[
            (oov_curve_df["field"] == field) & (oov_curve_df["src"] == src) & (oov_curve_df["head_size_N"] == (head_size or -1))
        ]
        oov_row_rate = None if oov_row.empty else oov_row.iloc[0]["oov_row_rate_valid"]
        oov_token_rate = None if oov_row.empty else oov_row.iloc[0]["oov_token_rate_valid"]
        new_value_rate = None if oov_row.empty else oov_row.iloc[0]["new_value_rate_valid_distinct"]
        trunc_row = trunc_df[(trunc_df["field"] == field) & (trunc_df["src"] == src) & (trunc_df["K"] == (max_len or -1))]
        retained_frac = None if trunc_row.empty else trunc_row.iloc[0]["retained_token_frac"]
        collision_row = hash_collision_df[
            (hash_collision_df["field"] == field)
            & (hash_collision_df["src"] == src)
            & (hash_collision_df["bucket"] == (hash_bucket or -1))
        ]
        collision_rate = None if collision_row.empty else collision_row.iloc[0]["collision_rate"]
        rationale_lines = [
            f"- coverage_rate={fmt(row.get('coverage_rate'))}, avg_nnz={fmt(row.get('avg_nnz_per_covered_row'))}, distinct_est={fmt(row.get('fid_distinct_est'),0)}",
            f"- OOV hint: encoding={encoding}, head_size={fmt(head_size,0)}",
            f"- Hash bucket hint: {fmt(hash_bucket,0)}",
        ]
        if max_len:
            rationale_lines.append(f"- Truncation: max_len={fmt(max_len,0)}, retained_token_frac@K={fmt(retained_frac)}")
        if psi_val is not None:
            rationale_lines.append(f"- Value PSI={fmt(psi_val)}")
        else:
            logger.warning("PSI missing for field=%s src=%s", field, src)
        if oov_row_rate is not None:
            rationale_lines.append(f"- OOV row@{fmt(head_size,0)}: {fmt(oov_row_rate)}, token@{fmt(head_size,0)}: {fmt(oov_token_rate)}, new_value:{fmt(new_value_rate)}")
        else:
            logger.warning("OOV metrics missing for field=%s src=%s", field, src)
        if collision_rate is not None:
            rationale_lines.append(f"- Collision@bucket {fmt(hash_bucket,0)}: {fmt(collision_rate)}")
        else:
            logger.warning("Collision metrics missing for field=%s src=%s", field, src)
        rationale_lines.append("- 风险/监控：OOV & 新值率、max_len p99、hash collision、PSI、nan/inf")
        rationales.append(f"### field {field} (src={src})\n" + "\n".join(rationale_lines) + "\n")

    patch = {
        "featuremap_version": (existing_fm.get("featuremap_version", 1) if existing_fm else 1) + 0,
        "dataset": existing_fm.get("dataset", "Ali-CCP") if existing_fm else "Ali-CCP",
        "generated_from": {
            "source": "eda-extra",
        },
        "defaults": existing_defaults,
        "features": features,
    }
    rationale_md = "\n".join(rationales)

    if existing_feat_map:
        diff_lines = ["|field|src|old|new|", "|---|---|---|---|"]
        for feat in features:
            key = (feat["field"], feat["src"])
            old = existing_feat_map.get(key)
            if not old:
                diff_lines.append(f"|{feat['field']}|{feat['src']}|<none>|new field|")
                continue
            changed = []
            for k in ["encoding", "vocab_size", "vocab_head_size", "hybrid_rule", "tail_encoding", "hash_bucket", "max_len", "embedding_dim", "pooling"]:
                ov = old.get(k)
                nv = feat.get(k)
                if ov != nv:
                    changed.append(f"{k}:{ov}->{nv}")
            if changed:
                diff_lines.append(f"|{feat['field']}|{feat['src']}|{old.get('encoding')}|{' ; '.join(changed)}|")
        diff_md = "\n".join(diff_lines)
    else:
        diff_md = "这是首版 patch（仓库中未找到旧的 featuremap 文件）。"

    return patch, rationale_md, diff_md


def write_featuremap_outputs(
    patch: Dict,
    rationale_md: str,
    diff_md: str,
    out_dir: Path,
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path = out_dir / "featuremap_patch.yaml"
    rationale_path = out_dir / "featuremap_rationale.md"
    diff_path = out_dir / "featuremap_diff.md"

    if patch_path.exists() and not overwrite:
        logger.info("Skip writing %s (exists)", patch_path)
    else:
        with open(patch_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(patch, f, allow_unicode=True, sort_keys=False)
    if rationale_path.exists() and not overwrite:
        logger.info("Skip writing %s (exists)", rationale_path)
    else:
        with open(rationale_path, "w", encoding="utf-8") as f:
            f.write(rationale_md)
    if diff_path.exists() and not overwrite:
        logger.info("Skip writing %s (exists)", diff_path)
    else:
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(diff_md)
