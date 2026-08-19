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
uv venv vein_module/.venv --python 3.12
uv pip install --python vein_module/.venv -r vein_module/requirements.txt

uv venv infection_module/.venv --python 3.12
uv pip install --python infection_module/.venv -r infection_module/requirements.txt
```

### Full containerised stack (deployment parity check)

```bash
docker compose --profile full up --build
```

---

## Data

```bash
python scripts/download_datasets.py --dataset all
```

Fetches CUBITAL (Google Drive, needs `gdown`), AZH (git clone), and Medetec
(scrape). U-DAVIS has no public mirror and must be requested from its authors.
Full detail, licensing caveats, and label caveats: [`docs/dataset_sources.md`](docs/dataset_sources.md).

`data/raw/` and `data/processed/` are git-ignored — reproducibility comes from
the split manifest written in Phase 2, not from committing images.

---

## Project status

- [x] **Phase 1** — Scaffolding: repo structure, envs, compose, CI-ready smoke test
- [ ] **Phase 2** — Data pipeline: subject-level splits, augmentation, data report
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
