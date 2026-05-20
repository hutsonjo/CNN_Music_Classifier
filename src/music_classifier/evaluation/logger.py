"""Lightweight experiment tracking via JSON logs.

Each training or evaluation run gets a timestamped entry with its config
and results. The entire log is in a single JSON file that's easy to
check into git and share across the team.

The log file is append-only during a session — each ``log_results`` call
rewrites the file with the updated list of runs, so concurrent writes
from multiple processes are not safe.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


class ExperimentLogger:
    """
    Tracks experiment configs and results in a JSON log file.
    """

    def __init__(self, log_dir: PathLike = "experiments") -> None:
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "experiment_log.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                self.runs: list[dict[str, Any]] = json.load(f)
        else:
            self.runs = []

    def _save(self) -> None:
        with open(self.log_file, "w") as f:
            json.dump(self.runs, f, indent=2, default=str)

    def start_run(self, config: dict[str, Any]) -> str:
        """Start a new run, record its config, and return a timestamp-based run ID."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": None,
            "status": "running",
        }
        self.runs.append(run)
        self._save()
        print(f"Started run: {run_id}")
        return run_id

    def log_results(self, run_id: str, results: dict[str, Any]) -> None:
        """Attach evaluation results to an existing run."""
        serializable: dict[str, Any] = {}
        skip_keys = {"y_pred", "y_probs"}  # large arrays — omit from log

        for k, v in results.items():
            if k in skip_keys:
                continue
            elif hasattr(v, "tolist"):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v

        for run in self.runs:
            if run["run_id"] == run_id:
                run["results"] = serializable
                run["status"] = "completed"
                break

        self._save()
        print(f"Results logged for run: {run_id}")

    def summary(self) -> None:
        """Print a table summarizing every logged run to stdout."""
        print(
            f"\n{'Run ID':<20} {'Status':<12} {'Accuracy':<10} "
            f"{'F1 (macro)':<12} {'Config'}"
        )
        print("-" * 90)
        for run in self.runs:
            r = run.get("results") or {}
            acc = f"{r['accuracy']:.4f}" if "accuracy" in r else "—"
            f1 = f"{r['f1_macro']:.4f}" if "f1_macro" in r else "—"

            cfg = run.get("config", {})
            cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
            if len(cfg_str) > 40:
                cfg_str = cfg_str[:37] + "..."

            print(
                f"{run['run_id']:<20} {run['status']:<12} {acc:<10} {f1:<12} {cfg_str}"
            )

    def get_best_run(self, metric: str = "accuracy") -> dict[str, Any] | None:
        """Return the run with the highest value of the given metric."""

        completed = [r for r in self.runs if r["status"] == "completed"]
        if not completed:
            print("No completed runs found.")
            return None
        return max(completed, key=lambda r: r["results"].get(metric, 0))