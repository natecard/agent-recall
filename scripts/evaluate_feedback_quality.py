#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_recall.core.retrieval_feedback import evaluate_feedback_impact
from agent_recall.storage.sqlite import SQLiteStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval ranking quality before/after feedback weighting."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(".agent/state.db"),
        help="Path to SQLite state database.",
    )
    parser.add_argument("--tenant-id", default="default", help="Tenant namespace.")
    parser.add_argument("--project-id", default="default", help="Project namespace.")
    parser.add_argument("--top-k", type=int, default=10, help="Ranking depth to score.")
    parser.add_argument(
        "--min-labels-per-query",
        type=int,
        default=2,
        help="Minimum positive/negative labels needed to evaluate a query.",
    )
    parser.add_argument(
        "--feedback-limit",
        type=int,
        default=2000,
        help="Maximum feedback rows to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage = SQLiteStorage(
        db_path=args.db_path,
        tenant_id=args.tenant_id,
        project_id=args.project_id,
    )
    report = evaluate_feedback_impact(
        storage,
        top_k=max(1, int(args.top_k)),
        min_labels_per_query=max(1, int(args.min_labels_per_query)),
        feedback_limit=max(1, int(args.feedback_limit)),
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
