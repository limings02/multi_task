from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

from src.core.logging import get_logger
from src.core.paths import find_repo_root, resolve_path
from src.eda.extra.common import (
    TokenDirs,
    load_config,
    locate_tokens_dirs,
    summarize_metadata,
    write_json,
    write_table,
)
from src.eda.extra.hash_collision import compute_hash_collision
from src.eda.extra.length import compute_length_metrics, plot_truncation_curves
from src.eda.extra.oov import compute_oov_curves
from src.eda.extra.psi import compute_val_drift, plot_val_psi
from src.eda.extra.featuremap_reco import build_featuremap_patch, write_featuremap_outputs


logger = get_logger(__name__)


def run_eda_extra(
    config_path: str,
    in_stats_dir: str,
    out_stats_dir: str,
    plots_dir: str,
    debug_sample: bool = False,
    overwrite: bool = False,
) -> Dict[str, str]:
    cfg_path = Path(config_path)
    repo_root = find_repo_root(cfg_path)
    config = load_config(cfg_path)
    token_dirs: TokenDirs = locate_tokens_dirs(repo_root, config, debug_sample=debug_sample)

    stats_dir = resolve_path(repo_root, Path(in_stats_dir))
    out_stats = resolve_path(repo_root, Path(out_stats_dir))
    plots_out = resolve_path(repo_root, Path(plots_dir))
    out_stats.mkdir(parents=True, exist_ok=True)
    plots_out.mkdir(parents=True, exist_ok=True)

    # Load base stats
    field_stats_df = pd.read_parquet(stats_dir / "field_stats_train.parquet")
    field_topk_df = pd.read_parquet(stats_dir / "field_topk_train.parquet")
    samples_train = json.load(open(stats_dir / "samples_stats_train.json", "r", encoding="utf-8"))
    samples_valid = json.load(open(stats_dir / "samples_stats_valid.json", "r", encoding="utf-8"))
    train_rows = int(samples_train.get("rows", 0))
    valid_rows = int(samples_valid.get("rows", 0))

    con = duckdb.connect()
    # OOV
    oov_curve_path = out_stats / "oov_curve.parquet"
    oov_summary_path = out_stats / "oov_summary.parquet"
    train_topn_cache = out_stats / "train_topn_200k.parquet"
    oov_curve_tbl, oov_summary_tbl = compute_oov_curves(
        con=con,
        train_tokens=token_dirs.train,
        valid_tokens=token_dirs.valid,
        cache_topn_path=train_topn_cache,
        out_curve_path=oov_curve_path,
        out_summary_path=oov_summary_path,
        overwrite=overwrite,
    )

    # Length / truncation
    quantiles_path = out_stats / "field_length_quantiles.parquet"
    trunc_path = out_stats / "truncation_loss_curve.parquet"
    quant_tbl, trunc_tbl = compute_length_metrics(
        con=con,
        train_tokens=token_dirs.train,
        field_stats_df=field_stats_df,
        train_rows=train_rows,
        out_quantile_path=quantiles_path,
        out_trunc_path=trunc_path,
        overwrite=overwrite,
    )

    # Hash collision
    hash_parquet_path = out_stats / "hash_collision_est.parquet"
    hash_json_path = out_stats / "hash_bucket_reco.json"
    hash_tbl = compute_hash_collision(
        field_stats_df=field_stats_df,
        out_parquet=hash_parquet_path,
        out_json=hash_json_path,
        overwrite=overwrite,
    )
    hash_reco_json = json.load(open(hash_json_path, "r", encoding="utf-8"))

    # Value PSI
    val_fields = [(r["field"], int(r["src"])) for _, r in field_stats_df[field_stats_df["val_non1_ratio"] > 0].iterrows()]
    psi_profile_path = out_stats / "val_profile_train_valid.parquet"
    psi_path = out_stats / "val_psi.parquet"
    psi_profile_tbl, psi_tbl = compute_val_drift(
        con=con,
        fields=val_fields,
        train_tokens=token_dirs.train,
        valid_tokens=token_dirs.valid,
        out_profile_path=psi_profile_path,
        out_psi_path=psi_path,
        overwrite=overwrite,
    )

    # Featuremap recommendations
    oov_summary_df = oov_summary_tbl.to_pandas()
    trunc_df = trunc_tbl.to_pandas()
    psi_df = psi_tbl.to_pandas()
    quant_df = quant_tbl.to_pandas()
    hash_collision_df = hash_tbl.to_pandas()
    oov_curve_df = oov_curve_tbl.to_pandas()
    patch, rationale_md, diff_md = build_featuremap_patch(
        field_stats_df=field_stats_df,
        oov_summary_df=oov_summary_df,
        oov_curve_df=oov_curve_df,
        quant_df=quant_df,
        trunc_df=trunc_df,
        hash_collision_df=hash_collision_df,
        hash_bucket_json=hash_reco_json,
        val_fields=[f for f, _ in val_fields],
        psi_df=psi_df,
        featuremap_path=repo_root / "configs" / "dataset" / "featuremap_v1.yaml",
    )
    write_featuremap_outputs(
        patch=patch,
        rationale_md=rationale_md,
        diff_md=diff_md,
        out_dir=plots_out,
        overwrite=overwrite,
    )

    # Plots
    plot_oov_path = plots_out / "fig_oov_curve_top_fields.png"
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if oov_curve_tbl.num_rows > 0:
            df_oov = oov_curve_df
            top_fields = (
                field_stats_df.assign(token_rows=field_stats_df.get("token_rows", field_stats_df["coverage_rate"]))
                .sort_values("token_rows", ascending=False)
                .head(6)
            )
            df_plot = df_oov[df_oov.apply(lambda r: (r["field"], r["src"]) in set(zip(top_fields["field"], top_fields["src"])), axis=1)]
            if not df_plot.empty:
                plt.figure(figsize=(10, 6))
                sns.lineplot(
                    data=df_plot,
                    x="head_size_N",
                    y="oov_token_rate_valid",
                    hue=df_plot["field"].astype(str) + "_s" + df_plot["src"].astype(str),
                    marker="o",
                )
                sns.lineplot(
                    data=df_plot,
                    x="head_size_N",
                    y="oov_row_rate_valid",
                    hue=df_plot["field"].astype(str) + "_s" + df_plot["src"].astype(str),
                    marker="x",
                    linestyle="--",
                    legend=False,
                )
                plt.xlabel("head size")
                plt.ylabel("OOV rate")
                plt.title("OOV curve (token vs row)")
                plt.tight_layout()
                plot_oov_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_oov_path, dpi=200)
                plt.close()
    except Exception as e:  # pragma: no cover
        logger.warning("Plotting OOV curve failed: %s", e)

    plot_truncation_curves(trunc_df, field_stats_df, plots_out / "fig_trunc_curve_top_multihot.png")
    plot_val_psi(psi_df, plots_out / "fig_val_psi_top_fields.png")

    # Metadata
    metadata = summarize_metadata(
        config_path=cfg_path,
        in_stats=stats_dir,
        token_dirs=token_dirs,
        sample_rows_train=train_rows,
        sample_rows_valid=valid_rows,
        repo_root=repo_root,
        debug_sample=debug_sample,
    )
    write_json(out_stats / "metadata.json", metadata, overwrite=True)
    write_json(plots_out / "metadata.json", metadata, overwrite=True)

    return {
        "oov_curve": str(oov_curve_path),
        "oov_summary": str(oov_summary_path),
        "length_quantiles": str(quantiles_path),
        "truncation_loss_curve": str(trunc_path),
        "hash_collision": str(hash_parquet_path),
        "hash_bucket_reco": str(hash_json_path),
        "val_profile": str(psi_profile_path),
        "val_psi": str(psi_path),
        "featuremap_patch": str(plots_out / "featuremap_patch.yaml"),
    }
