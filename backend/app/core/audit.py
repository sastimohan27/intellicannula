"""Structured audit logging for every ML call.

This doubles as the clinical audit trail, so it records what would be needed to
reconstruct a recommendation after the fact: which patient, which endpoint,
which model version, the hash of the exact image, the output, and how long it
took. See docs/audit_logging.md.

Deliberately not recorded: the image bytes themselves, or any free-text patient
identifier. The hash links to the stored object without duplicating PHI into
the log stream.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone

audit_log = logging.getLogger("intellicannula.audit")


@contextmanager
def audit_ml_call(endpoint: str, patient_id: int, model_version: str, image_sha256: str):
    """Time an ML call and emit one structured audit record.

    Records failures as well as successes - an endpoint that errored on a
    patient's image is exactly the kind of event an audit needs to show.
    """
    started = time.perf_counter()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "patient_id": patient_id,
        "model_version": model_version,
        "image_sha256": image_sha256,
    }
    result: dict = {}
    try:
        yield result
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - re-raised below
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        record["output"] = result
        audit_log.info(json.dumps(record, default=str))
