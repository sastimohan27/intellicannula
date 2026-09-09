# Deployment Checklist

**Nothing here is a clearance to deploy.** This is a research prototype. Real
clinical use requires IRB approval and a formal validation study that this
project has not begun.

## Before any clinical exposure

- [ ] IRB approval for the specific deployment context
- [ ] Prospective validation on the target population — the vein model is
      trained on subjects aged 5–22 (median 11), 97% slim; the clinical need is
      elderly and obese adults
- [ ] Infection model retrained on real labelled IV sites, not the current
      wound-versus-skin proxy
- [ ] Cannula detector trained (100–200 labelled images, currently absent)
- [ ] Clinical lead sign-off on alert thresholds, which are currently plan
      values never fitted to outcome data
- [ ] Data protection assessment — patient images are identifiable
- [ ] The known off-patient prediction failure fixed and re-evaluated

## Model versioning

Every prediction stores `model_version` alongside it (`vein_scans.model_version`,
`site_checkins.model_version`). Rules:

1. **Bump the version string on every retrain**, in `backend/app/core/config.py`.
2. Never overwrite a checkpoint in place — a silent swap makes historical
   predictions unreproducible.
3. `GET /models` reports live versions and whether weights are present.
4. Keep the `training_config.json` written beside each checkpoint; it records
   hyperparameters, the measured `pos_weight`, and the data split.

## ONNX export

`vein_module/src/export.py` and `infection_module/src/export.py` export and
**verify numerical parity** — the vein model currently agrees with PyTorch to
6.6e-06 against a 1e-04 tolerance. Treat a verification failure as blocking: an
export that computes different numbers changes clinical recommendations.

## Known issues that must be resolved before scaling

**Single-process OpenMP conflict.** PyTorch and LightGBM each bundle an OpenMP
runtime; loading both in one process segfaults on macOS. Currently mitigated by
pinning both to one OpenMP thread. **The correct fix is to run the vein and
infection modules as separate services** — the per-module Dockerfiles exist for
this. Do that before scaling, rather than relying on the thread pinning.

**In-process WebSocket fan-out.** `/ws/alerts` keeps connections in a
per-process set. Correct for one worker; with several, a clinician connected to
worker B misses an alert raised on worker A. Multi-worker deployment needs a
shared broker (Redis pub/sub).

**`create_all` at startup.** Adequate for a prototype. Use the Alembic
migrations for anything real, so schema changes are versioned.

## Scaling

- Both ML services scale horizontally; they are stateless per request
- Latency measured at vein-scan ~654 ms, site-check ~7 ms (CPU, single image)
- The image store already abstracts S3 versus local disk
- Postgres needs connection pooling sized to worker count
- GPU is optional — CPU latency is already inside bedside tolerance

## Monitoring

- Audit log volume and error rate by endpoint (see `audit_logging.md`)
- p95 latency per pipeline
- Rate of `registration_degraded` — a rise means capture quality is dropping
- Rate of vein-scan refusals — a rise means image quality or population drift
- Alert rate per ward — a spike is more likely a model or camera problem than an
  outbreak
