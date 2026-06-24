from __future__ import annotations

import argparse

from .index import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Index PMC-Patients notes into the chart namespace")
    ap.add_argument("--limit", type=int, default=None, help="cap notes (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="load + embed but skip the gateway")
    args = ap.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
