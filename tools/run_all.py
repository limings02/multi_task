"""
tools/run_all_simple.py

目标：
- 不做日志重定向
- 按顺序串行跑 3 个训练任务
- 终端里直接看到训练输出（和你手敲命令一样）
- 任何一个任务失败：默认继续跑后面的（可改为失败就停）
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Job:
    name: str
    config: str


def run_job(job: Job, python_exe: str) -> int:
    """
    串行执行一个 job，stdout/stderr 继承父进程（不重定向）。
    所以你会在终端里实时看到训练日志。
    """
    cmd = [
        python_exe,
        "-m", "src.cli.main",
        "train",
        "--config", job.config,
    ]

    print("\n" + "=" * 80)
    print(f"[RUN] {job.name}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)

    # 注意：不传 stdout/stderr => 默认继承当前终端
    # cwd 设为当前工作目录（建议你从项目根目录运行）
    rc = subprocess.call(cmd, cwd=os.getcwd())

    if rc == 0:
        print(f"[OK]  {job.name}")
    else:
        print(f"[FAIL] {job.name} (exit_code={rc})")
    return rc


def main() -> int:
    # 用当前解释器，确保就是你 conda env(test) 的 python
    python_exe = sys.executable

    jobs: List[Job] = [
        Job("job1_sharedbottom_ctr_cvr", "configs/experiments/deepfm_sharedbottom_train.yaml"),
        Job("job2_sharedbottom_ctr_only", "configs/experiments/deepfm_sharedbottom_train_ctr.yaml"),
        Job("job3_sharedbottom_cvr_only", "configs/experiments/deepfm_sharedbottom_train_cvr.yaml"),
    ]

    print(f"[INFO] python = {python_exe}")
    print(f"[INFO] jobs = {len(jobs)}")
    print("[INFO] Running sequentially (one finishes -> next starts).")

    any_fail = False
    for i, job in enumerate(jobs, 1):
        print(f"\n[QUEUE] ({i}/{len(jobs)}) {job.name}")
        rc = run_job(job, python_exe=python_exe)

        # 默认：失败也继续跑后面的（更适合夜间挂跑）
        if rc != 0:
            any_fail = True
            # 如果你想“失败就停”，把下面 2 行取消注释：
            # print("[STOP] stop on first failure.")
            # return rc

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
