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
        │  → pick site   │      │  → register     │
        │  → angle       │      │  → risk + trend │
        └────────────────┘      └─────────────────┘
                     ┌───────────────┴──────────────┐
                     │   PostgreSQL  +  S3/MinIO    │
                     │  sessions · history · audit  │
                     └──────────────────────────────┘
```

**Pre-cannulation (vein module)** — given a forearm image, segment subcutaneous
veins with an Attention U-Net, pick the widest straightest accessible segment as
an insertion site, and suggest a needle angle.

**Post-cannulation (infection module)** — given sequential photos of the same IV
site, detect the cannula hub with YOLOv11, register each new image against the
patient's baseline, extract colour/texture features, and flag deterioration
before it becomes phlebitis or cellulitis.

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

```bash
uv venv backend/.venv --python 3.12
uv pip install --python backend/.venv -r backend/requirements.txt
cd backend && ../backend/.venv/bin/uvicorn app.main:app --reload
```

`http://localhost:8000/health` → `{"status":"ok","version":"0.1.0"}`

### 3. Frontend

```bash
cd frontend && npm run dev    # http://localhost:3000
```

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
infection_module/.venv/bin/python -m pytest -q   # module tests (47)
cd backend && .venv/bin/python -m pytest -q      # API tests (2)
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

- [x] **Phase 1** — Scaffolding: repo structure, envs, compose, CI smoke test
- [x] **Phase 2** — Data pipeline: subject-level splits, augmentation, data report
      *(infection side complete on real data; vein side written and unit-tested
      but unrun — CUBITAL download is blocked, see below)*
- [ ] **Phase 3** — Vein module: Attention U-Net, site selector, angle heuristic
- [ ] **Phase 4** — Cannula detection: YOLOv11 on hand-labelled hub images
- [ ] **Phase 5** — Infection monitoring: registration, features, risk model, tracker
- [ ] **Phase 6** — Backend API + dashboard integration
- [ ] **Phase 7** — Evaluation & validation
- [ ] **Phase 8** — Deployment: ONNX export, audit logging, deployment checklist

## Known gaps

1. **No public longitudinal IV-site dataset** — the temporal tracker (Phase 5)
   has nothing real to validate against; a small IRB-approved pilot collection
   is the only real fix.
2. **No public cannula-detection dataset** — Phase 4 needs ~100–200 manually
   labelled images. Budget 1–2 days before starting it.
3. **No RGB vein data** — see "Camera modality" above.
4. **AZH has no infection labels** — only wound type. The Phase 5 `is_infected`
   target is a proxy label and every metric derived from it must say so.

## Scope

Research prototype. Not a medical device, not validated, not for clinical use.
Any real deployment requires IRB approval and a formal validation study.
