"""Pipeline CLI: `python -m pipeline.cli <command>`.

One subcommand per stage, plus the inspection commands. Uses argparse so `init`,
`stats`, and `queue` run before any of requirements.txt is installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import db
from .config import config


def _report(name: str, counts: dict) -> int:
    print(name + ": " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def _open(args: argparse.Namespace, *, read_only: bool = False):
    return db.connect(args.db, read_only=read_only)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    conn = db.init_db(args.db)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    print(f"initialised {args.db} (schema v{version})")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = _open(args, read_only=True)
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
    if counts["scored"]:
        surviving = counts["scored"] - counts["rejected"]
        print(f"\nsurvival rate: {surviving / counts['scored']:.1%} of scored shots")
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    from .stages import select_render

    conn = _open(args, read_only=not args.stage)
    try:
        if args.stage:
            rows = select_render.stage_queue(conn, args.limit)
            proposed = [(row, "queued") for row in rows]
        else:
            proposed = select_render.propose_queue(conn, args.limit)

        if not proposed:
            print("selection queue is empty")
            return 0
        for row, reason in proposed:
            moods = ", ".join(row["mood_tags"] or []) or "-"
            print(
                f"{row['id']:>6}  {(row['country'] or '?'):<14} {row['duration']:>5.1f}s  "
                f"{moods:<28} {reason:<32} {row['description'] or ''}"
            )
    finally:
        conn.close()
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    from .stages import select_render

    conn = _open(args)
    try:
        count = select_render.approve(conn, args.shot_ids)
    finally:
        conn.close()
    print(f"approved {count} shot(s)")
    return 0


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    from .stages import ingest

    conn = _open(args)
    try:
        return _report("ingest", ingest.run(conn, config, force=args.force, limit=args.limit))
    finally:
        conn.close()


def cmd_segment(args: argparse.Namespace) -> int:
    from .stages import scene_detect

    conn = _open(args)
    try:
        return _report("segment", scene_detect.run(conn, config, limit=args.limit))
    finally:
        conn.close()


def cmd_score(args: argparse.Namespace) -> int:
    from .stages import score

    conn = _open(args)
    try:
        if args.rejections_only:
            return _report("reject", score.apply_rejections(conn, config, dry_run=args.dry_run))
        return _report("score", score.run(conn, config, limit=args.limit))
    finally:
        conn.close()


def cmd_tag(args: argparse.Namespace) -> int:
    from .stages import tag

    conn = _open(args)
    try:
        return _report("tag", tag.run(conn, config, limit=args.limit))
    finally:
        conn.close()


def cmd_render(args: argparse.Namespace) -> int:
    from .stages import select_render

    conn = _open(args)
    try:
        profiles = tuple(args.profiles)
        return _report("render", select_render.run(conn, config, limit=args.limit, profiles=profiles))
    finally:
        conn.close()


def cmd_publish(args: argparse.Namespace) -> int:
    from .stages import distribute

    conn = _open(args)
    try:
        if args.sync_only:
            return _report("sync", distribute.sync_delivery(conn, config))
        return _report(
            "publish",
            distribute.run(
                conn, config, limit=args.limit, platforms=config.social_platforms
            ),
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    parser.add_argument("--db", default=str(config.db_path), help="path to the pipeline database")
    parser.add_argument("-v", "--verbose", action="store_true", help="log each item processed")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the database and apply the schema").set_defaults(
        func=cmd_init
    )

    stats = sub.add_parser("stats", help="show how far the archive has moved")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=cmd_stats)

    ingest = sub.add_parser("ingest", help="stage 1a: register source files")
    ingest.add_argument("--force", action="store_true", help="re-probe unchanged files")
    ingest.add_argument("--limit", type=int)
    ingest.set_defaults(func=cmd_ingest)

    segment = sub.add_parser("segment", help="stage 1b: split sources into shots")
    segment.add_argument("--limit", type=int)
    segment.set_defaults(func=cmd_segment)

    score = sub.add_parser("score", help="stage 2: score shots and cut the bottom percentile")
    score.add_argument("--limit", type=int)
    score.add_argument(
        "--rejections-only", action="store_true",
        help="re-draw the bar over existing scores without re-decoding frames",
    )
    score.add_argument("--dry-run", action="store_true", help="with --rejections-only, don't write")
    score.set_defaults(func=cmd_score)

    tag = sub.add_parser("tag", help="stage 3: describe surviving shots with a VLM")
    tag.add_argument("--limit", type=int)
    tag.set_defaults(func=cmd_tag)

    queue = sub.add_parser("queue", help="stage 4: show shots proposed for review")
    queue.add_argument("--limit", type=int, default=10)
    queue.add_argument("--stage", action="store_true", help="mark the proposals as queued")
    queue.set_defaults(func=cmd_queue)

    approve = sub.add_parser("approve", help="stage 4: approve shots for rendering")
    approve.add_argument("shot_ids", type=int, nargs="+")
    approve.set_defaults(func=cmd_approve)

    render = sub.add_parser("render", help="stage 4: render approved shots")
    render.add_argument("--limit", type=int)
    render.add_argument(
        "--profiles", nargs="+", default=["social", "licensing"],
        choices=["social", "licensing"],
    )
    render.set_defaults(func=cmd_render)

    publish = sub.add_parser("publish", help="stage 5: post via Zernio, then check delivery")
    publish.add_argument("--limit", type=int)
    publish.add_argument(
        "--sync-only", action="store_true",
        help="re-check delivery status of already-posted shots, publish nothing",
    )
    publish.set_defaults(func=cmd_publish)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
