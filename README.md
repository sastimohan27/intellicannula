# IntelliCannula

Two-module clinical decision support for peripheral IV cannulation. The system
**recommends, never acts** — every output is advisory and requires clinician
confirmation.

```
                     ┌──────────────────────────────┐
                     │   Frontend Dashboard (Next)  │
                     │  live capture · alerts · trend│
                     └───────────────┬──────────────┘
                                     │ REST / WebSocket
                     ┌───────────────┴──────────────┐
                     │      Backend API (FastAPI)   │
                     ├──────────────┬───────────────┤
        ┌────────────┴───┐      ┌───┴────────────┐
        │ Pre-cannulation│      │ Post-cannulation│
        │  VEIN MODULE   │      │ INFECTION MODULE│
        │                │      │                 │
        │ NIR image      │      │ RGB image       │
        │  → segment     │      │  → detect hub   │
        │  → pick site   │      │  → skin ring    │
        │  → angle       │      │  → redness+trend│
        └────────────────┘      └─────────────────┘
                     ┌───────────────┴──────────────┐
                     │   PostgreSQL  +  S3/MinIO    │
                     │  sessions · history · audit  │
                     └──────────────────────────────┘
```

**Pre-cannulation (vein module)** — given a forearm image, segment subcutaneous
veins with an Attention U-Net, pick the widest straightest accessible segment as
an insertion site, and suggest a needle angle.

**Post-cannulation (infection module)** — given a daily photo of the IV site,
check photo quality, detect the cannula hub with YOLOv11 (no hub → retake, never
a guess), measure redness (CIELAB Δa\*) in a ring of skin around the hub against
normal skin further out in the same photo, and compare with the patient's own
day-0 photo. Alerts fire on high redness, a sharp rise, or three rises in a row.
See [`infection_module/src/erythema.py`](infection_module/src/erythema.py) for
why this replaced the LightGBM proxy model in the live path.

---

## Camera modality — DECISION PENDING

This is the highest-leverage open question in the project and it blocks Phase 3.

Every public vein dataset with segmentation masks (CUBITAL, U-DAVIS) is
**near-infrared**. No public RGB vein dataset with masks exists. Meanwhile the
infection module needs **RGB**, because erythema is a colour signal that NIR
discards.

See the "Camera modality" section of the project notes for the full trade-off.
Once chosen, record it here — it determines `build_model_input()` in
`vein_module/src/preprocessing.py` and everything downstream.

---

## Runtime

Chosen setup: **native Python via `uv` for backend and ML, Docker for
infrastructure only.** This keeps Apple Silicon MPS acceleration available to
PyTorch, which containerised training on macOS would lose.

### 1. Infrastructure

```bash
cp .env.example .env          # fill in secrets
docker compose up -d          # postgres only
```

### 2. Backend

The API needs the ML dependencies (torch, OpenCV, ultralytics), which live in
`infection_module/.venv`. Install the API requirements into it and run from there:

```bash
uv pip install --python infection_module/.venv -r backend/requirements.txt
cd backend && ../infection_module/.venv/bin/python -m uvicorn app.main:app \
    --reload --host 0.0.0.0 --port 8001
```

`http://localhost:8001/health` → `{"status":"ok","version":"0.1.0"}`.
Run from `backend/.venv` instead and the ML endpoints return 503 naming the
missing package.

### 3. Frontend

```bash
cd frontend && npm run dev -- -H 0.0.0.0 -p 3000    # http://localhost:3000
```

### Using it from a phone

With both servers started with `--host 0.0.0.0` / `-H 0.0.0.0`, and the phone on
the same Wi-Fi, open `http://<laptop-IP>:3000` (`ipconfig getifaddr en0` on
macOS). Upload buttons open the rear camera directly. The frontend rewrites a
`localhost` API URL to the laptop's address, CORS allows private-network
origins, and `next.config.ts` allows the laptop's LAN IPs as dev origins.
Restart the frontend after changing networks.

### Demo images

`scripts/make_demo_sequence.py` builds a **simulated** three-day site sequence
(drawn hub on a real limb photo, redness added on days 2-3). It demonstrates the
pipeline end to end; it is not evidence of detecting real phlebitis.

### 4. ML modules (per-module venvs — torch is heavy, keep it out of the API env)

```bash
brew install libomp        # lightgbm links against the OpenMP runtime

uv venv vein_module/.venv --python 3.12
uv pip install --python vein_module/.venv -r vein_module/requirements.txt

uv venv infection_module/.venv --python 3.12
uv pip install --python infection_module/.venv -r infection_module/requirements.txt
```

### Tests

```bash
infection_module/.venv/bin/python -m pytest -q                    # module tests (188)
cd backend && ../infection_module/.venv/bin/python -m pytest -q   # API tests (39)
```

### Full containerised stack (deployment parity check)

```bash
docker compose --profile full up --build
```

---

## Data

### Download

```bash
python scripts/download_datasets.py --dataset azh       # git clone, ~58 MB
python scripts/download_datasets.py --dataset medetec   # full-size gallery scrape
python scripts/download_datasets.py --dataset cubital   # needs `uv tool install gdown`
```

| Dataset | Status |
|---|---|
| AZH | ✅ 930 images, 6 classes |
| Medetec | ✅ full-size wound galleries |
| CUBITAL | ⛔ blocked by a Fortinet TLS-inspection appliance on this network |
| U-DAVIS | ⚠️ no public mirror — request from the authors |

The CUBITAL blocker and the three ways around it are documented in
[`docs/dataset_sources.md`](docs/dataset_sources.md). The script honours
`SSL_CERT_FILE` if you have the appliance's CA.

### Prepare

```bash
python infection_module/src/data_loader.py --prepare
python vein_module/src/data_loader.py --prepare      # once CUBITAL is present
python scripts/data_report.py                        # -> docs/data_report.md
```

`data/raw/` and `data/processed/` are git-ignored — reproducibility comes from
the `split_manifest.json` each preparer writes, not from committing images.

### Label honesty

No public dataset labels a wound as infected. The infection module's
`is_infected` target is a **proxy**: AZH wound classes → 1, normal skin → 0,
background dropped. It learns *wound tissue vs intact skin*. Every metric built
on it is reported as proxy-label performance, and the caveat is written into
`split_manifest.json` so it travels with the data.

---

## Project status

- [x] **Phase 1** — Scaffolding: repo structure, envs, compose, CI
- [x] **Phase 2** — Data pipeline: subject-level splits, augmentation, data report
- [x] **Phase 3** — Vein module: Attention U-Net (7.85M params), trained on
      CUBITAL. Test Dice 0.3698 — under-trained, see limitations
- [x] **Phase 4** — Cannula detection: trained via domain-randomized procedural
      compositing on real skin backgrounds; val mAP@50 0.995, precision 0.982
- [x] **Phase 5** — Infection monitoring: registration, 30 features, LightGBM
      risk model, temporal tracker
- [x] **Phase 6** — Backend API + clinician dashboard, verified end to end
- [x] **Phase 7** — Evaluation: [`full_system_eval.md`](docs/full_system_eval.md),
      latency benchmark
- [x] **Phase 8** — ONNX export with parity check, audit logging, deployment
      checklist
- [x] **Phase 9** — Detection correctness (see
      [`docs/REVIEW2_RESULTS.md`](docs/REVIEW2_RESULTS.md)): vein model resumed
      to epoch 16, tuned mask post-processing (test Dice 0.370 → 0.567),
      strict cannula gate, vein grade fix
- [x] **Phase 10** — Redness-based site scoring replacing the saturating proxy
      model, photo-quality gate, phone capture, check-due reminder

### Results at a glance

| Component | Metric | Value |
|---|---|---|
| Vein segmentation | Dice / IoU (test, epoch 16, thr 0.9) | 0.567 / 0.413 (was 0.370 / 0.239) |
| Vein segmentation | Precision / recall (test) | 0.495 / 0.758 (was 0.262 / 0.814) |
| Cannula gate | No-cannula images passing (n=121) | 0 |
| Site redness | Functional tests (monotonic, tape/plastic excluded) | pass — **not clinical validation** |
| Infection risk (retired from API) | F1 (5-fold, grouped) | 0.9885 — **proxy labels** |
| Latency | vein-scan / site-check | 654 ms / 7 ms |
| ONNX parity | max abs difference | 6.6e-06 |
| Tests | module + API | 188 + 39 |

Read [`docs/full_system_eval.md`](docs/full_system_eval.md) before quoting any
of these. The infection F1 is a proxy-label number that an ablation shows is
driven by colour heterogeneity rather than erythema, and the vein model has a
known failure where it can mark an insertion point off the patient.

## Known gaps

1. **No public longitudinal IV-site dataset** — the temporal tracker (Phase 5)
   has nothing real to validate against; a small IRB-approved pilot collection
   is the only real fix.
2. **No public cannula-detection dataset** — Phase 4 needs ~100–200 manually
   labelled images. Budget 1–2 days before starting it.
3. **No RGB vein data** — see "Camera modality" above.
4. **AZH has no infection labels** — only wound type. The Phase 5 `is_infected`
   target is a proxy label and every metric derived from it must say so. It
   saturated at ~0.999 on real IV photos, so the API now uses the redness
   measurement instead.
5. **Redness thresholds are not clinically validated** — they need daily
   nurse-scored (VIP) photos of real IV sites to calibrate.
6. **Cannula detector trained on 90 synthetic images** — misses some real hubs
   (the gate turns a miss into a retake request, not a wrong score).

## Scope

Research prototype. Not a medical device, not validated, not for clinical use.
Any real deployment requires IRB approval and a formal validation study.
