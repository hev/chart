from __future__ import annotations

import argparse
from pathlib import Path

from chart_common.cli import positive_int

from .index import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Index PMC-Patients notes into the chart namespace")
    ap.add_argument("--limit", type=positive_int, default=None, help="cap notes (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="load + embed but skip the gateway")
    ap.add_argument("--out", type=Path, default=None, help="write the index report to this JSON path")
    args = ap.parse_args()
    main(limit=args.limit, dry_run=args.dry_run, out=args.out)
