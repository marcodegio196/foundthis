"""One module per pipeline stage.

Every stage exposes `run(conn, cfg, **options) -> dict` and communicates only
through the database. No stage imports another.
"""
