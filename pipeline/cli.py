"""Pipeline CLI: `python -m pipeline.cli <command>`.

Stage runners get wired in here as they land; for now this covers creating the
database and inspecting its state. Uses argparse so it runs before any of
requirements.txt is installed.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import db
from .config import config


def cmd_init(args: argparse.Namespace) -> int:
    conn = db.init_db(args.db)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    print(f"initialised {args.db} (schema v{version})")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    try:
        counts = db.stats(conn)
    finally:
        conn.close()
    if args.json:
        print(json.dumps(counts, indent=2))
        return 0

    width = max(len(k) for k in counts)
    for key, value in counts.items():
        print(f"{key.replace('_', ' '):<{width}}  {value:>8,}")

    surviving = counts["scored"] - counts["rejected"]
    if counts["scored"]:
        print(f"\nsurvival rate: {surviving / counts['scored']:.1%} of scored shots")
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    conn = db.connect(args.db, read_only=True)
    try:
        rows = db.selection_queue(conn, limit=args.limit)
    finally:
        conn.close()
    if not rows:
        print("selection queue is empty")
        return 0
    for row in rows:
        moods = ", ".join(row["mood_tags"] or []) or "-"
        print(
            f"{row['id']:>6}  {(row['country'] or '?'):<14} {row['duration']:>5.1f}s  "
            f"{moods:<32} {row['description'] or ''}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    parser.add_argument(
        "--db", default=str(config.db_path), help="path to the pipeline database"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and apply the schema").set_defaults(
        func=cmd_init
    )

    stats = sub.add_parser("stats", help="show how far the archive has moved")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=cmd_stats)

    queue = sub.add_parser("queue", help="show shots awaiting a stage 4 decision")
    queue.add_argument("--limit", type=int, default=25)
    queue.set_defaults(func=cmd_queue)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
