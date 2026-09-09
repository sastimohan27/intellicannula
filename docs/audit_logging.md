# Audit Logging

Every ML call writes one structured record. This is the clinical audit trail,
not debug output — it exists so that a recommendation a clinician acted on can
be reconstructed afterwards.

## Schema

Emitted as JSON on the `intellicannula.audit` logger by
`backend/app/core/audit.py`:

```json
{
  "timestamp": "2026-09-09T03:28:17.516830+00:00",
  "endpoint": "site-check",
  "patient_id": 1,
  "model_version": "lgbm-risk-v1",
  "image_sha256": "24428ba50550212...",
  "status": "ok",
  "latency_ms": 73.55,
  "output": {"risk_score": 0.998, "registration_degraded": false}
}
```

| Field | Why it is here |
|---|---|
| `timestamp` | UTC, ISO 8601. Ordering across sites and time zones. |
| `endpoint` | Which pipeline ran. |
| `patient_id` | Links to the record, without embedding a name. |
| `model_version` | **The critical field.** Models get retrained; behaviour changes. Without this you cannot say which model produced a given recommendation. |
| `image_sha256` | Proves exactly which image was analysed, and detects re-submission, without copying the image into the log. |
| `status` | `ok` or `error`. |
| `error` | Exception type and message when it failed. |
| `latency_ms` | Performance, and evidence of degradation over time. |
| `output` | The recommendation itself. |

## What is deliberately absent

- **Image bytes.** The hash links to the stored object; duplicating pixels into
  a log stream multiplies where PHI lives.
- **Patient names or MRNs.** `patient_id` resolves through the database, which
  has its own access controls. Log aggregators frequently do not.

## Failures are recorded too

An endpoint that errored on a patient's image is exactly what an audit needs to
show. `audit_ml_call` records the failure and re-raises, so the trail contains
attempts, not just successes. This is how the ROI coordinate-frame bug was
found during end-to-end testing — the audit record carried the precise message
(`ROI [140.0, 91.5, 420.0, 274.5] is empty on a (80, 94) image`) before anyone
opened a debugger.

## How this maps to "clinician confirmation required"

The system is advisory. Nothing it produces takes effect on its own:

1. A clinician uploads an image.
2. The system returns a recommendation **plus its confidence and its reason** —
   including refusing outright when no site meets the bar.
3. The clinician decides. The UI labels every output advisory and shows the
   model version.
4. The audit record captures what was recommended, by which model version, on
   which image, at what time.

There is no endpoint that performs an action on a patient. The only writes are
to the system's own records.

## Retention and access — not implemented

This prototype writes to stdout. A real deployment needs: append-only storage,
a defined retention period matching local record-keeping law, access controls
and access logging on the audit store itself, and tamper-evidence. None of that
is built here, and none of it should be assumed.
