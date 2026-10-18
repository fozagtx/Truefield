# Truefield: deployment handoff

Truefield is a maintenance copilot for factory technicians. It detects an early bearing fault from vibration data (an autoencoder plus physics features), has an LLM explain it with citations, drafts a lockout/tagout-safe work order, sends risky cases to a supervisor, and closes the job from a photo, PDF or voice note.

## What's real vs. what needs config

| Part | Works out of the box | Needs your config |
|---|---|---|
| Features, rules engine, autoencoder, work-order / approval / close-out flow, UI, SQLite | ✅ (CPU) | – |
| LLM diagnosis + WO drafting | – (**required**; the API won't start without it) | `MODEL_ACCESS_KEY` → live `gpt-4o` at `https://inference.do-ai.run/v1` |
| Evolus library, workflow, extractor, AI Tool `minutes` | skipped, shown as `not configured` (WO drafted by the LLM, close-out fields entered manually) | Evolus IDs in `.env` |
| Vibration data | synthetic fallback (labelled) | real NASA IMS via `scripts/download_ims.sh` |
| Temperature channel | **always synthetic** (labelled) | – |
| GPU training / benchmarks | CPU numbers only | AMD MI300X droplet |

## Prerequisites

- Docker with **Compose ≥ 2.24** (`docker compose version`), **or** Python 3.11+.
- Optional: an Evolus event workspace, an AMD Developer Cloud MI300X droplet, and a Hugging Face token (for Gemma).

## Deploy

```bash
git clone https://github.com/fozagtx/Truefield && cd Truefield
cp .env.example .env            # set MODEL_ACCESS_KEY (+ Evolus IDs if you have them); never commit .env
docker compose up --build -d    # API http://localhost:8080 (docs: /docs) · UI http://localhost:8501
```

Without Docker:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
set -a && . ./.env && set +a
uvicorn truefield.api.main:app --host 0.0.0.0 --port 8080 &
TRUEFIELD_API_URL=http://localhost:8080 streamlit run ui/app.py --server.port 8501
```

On first start the API trains the autoencoder on CPU (seconds) and saves it to `models/ae-v1.pt`.

## Verify

```bash
curl -s localhost:8080/health          # modes.llm = "do", modes.llm_model = "gpt-4o"; Evolus blocks "live" or "not configured"
curl -s -X POST "localhost:8080/replay/start?speed=50&start_frac=0.6"
sleep 10 && curl -s localhost:8080/wo  # one work order, status "pending_approval", drafted_by "evolus_workflow" or "agent_direct:do (Evolus workflow not configured)"
curl -s localhost:8080/audit           # every LLM / Evolus call, with its status and latency
```

Tests: `MOCK=1 pytest -q` (56 tests, no network). `MOCK=1` is the **only** way to get mock responses; never set it in a real deployment.

## Evolus checklist

Create the workspace **from the event link** (evolus.ai/en/hackathon), then:

- [ ] API key → `EVOLUS_API_KEY`; the space ID shown on the API keys page → `EVOLUS_APP_ID`
- [ ] Knowledge library `truefield-kb` → `EVOLUS_LIBRARY_ID`; run `python scripts/seed_library.py` and wait until every document is indexed
- [ ] Agent `Truefield Diagnostician` (prompt: `truefield/prompts/diagnostician.txt`, temperature 0.2, library attached) → `EVOLUS_AGENT_ID`
- [ ] Workflow `Draft Work Order`: API trigger, params `asset_id` + `route`, an agent step with `truefield/prompts/draft_wo.txt`; **turn it ON** → `EVOLUS_WORKFLOW_ID`
- [ ] Extractor `Close-out form` (fields from `truefield/schemas/closeout.json`) → `EVOLUS_EXTRACTOR_ID`
- [ ] To route diagnosis through the Evolus agent instead of gpt-4o, set `LLM_PROVIDER=evolus`
- [ ] Confirm the shapes marked `TODO: confirm in workspace API reference` in `truefield/evolus/*.py`:
  - the workflow execute `parameters` field name, and the output field in an execution or callback;
  - the extractor's owning agent (assumed to be `EVOLUS_AGENT_ID`) and the `ask-file` response;
  - the `import-url` body and the documents-list envelope;
  - the AI Tool `run` body fields.

## AMD notes

- **Anomaly model on ROCm:** run `./deploy/train_rocm.sh` on the droplet. It uses `rocm/pytorch`, logs the GPU name, trains `models/ae-v1.pt` and runs `bench/ae_bench.py`. Then run `python bench/collect_results.py` to refresh the Bench tab.
- **Optional Gemma on vLLM:** `HF_TOKEN=... ./deploy/serve_gemma.sh` prints a `VLLM_API_KEY`. Use it with `LLM_PROVIDER=vllm VLLM_BASE_URL=http://<ip>:8000/v1 VLLM_API_KEY=...`, or register it as an Evolus model. Benchmark with `./bench/vllm_bench.sh`.
- ⚠️ **gpt-4o does not run on AMD.** Prize eligibility for AMD (meaningful use), Gemma and Evolus ("agent on an open model via your own vLLM on AMD") requires an **open model served on AMD**. For the submission, serve Gemma via `deploy/serve_gemma.sh` and use it (directly or as the Evolus agent's model). Keep gpt-4o as a fallback only.

## Real data

```bash
sudo apt-get install -y p7zip-full unrar     # for the nested 7z/rar archives
DATA_DIR=./data ./scripts/download_ims.sh    # 1.08 GB → data/ims/2nd_test (984 files)
python scripts/make_demo_slice.py --stride 2 # → data/demo/ims_set2_b1.npz (picked up automatically)
rm -f models/ae-v1.pt                        # retrain on the real data at next start (or ./deploy/train_rocm.sh)
python scripts/eval_leadtime.py              # real lead-time numbers vs a naive 2×RMS alarm
```

Data and weights are git-ignored; never commit them.

## Submission checklist

- [ ] Repo public, MIT `LICENSE` present
- [ ] `docker compose up` works from a clean clone; README quickstart runs
- [ ] No keys in git (`git grep -nIi "key\|token"` shows only blanks/placeholders); `.env` untracked
- [ ] Open model on AMD in use (see AMD notes); `bench/results.json` has MI300X numbers
- [ ] Demo video (≈3 min), pitch deck, cover image
- [ ] lablab form: tracks, tags, GitHub URL, demo URL

## Troubleshooting

| Symptom | Fix |
|---|---|
| API exits at startup: `MODEL_ACCESS_KEY is not set. Set it in .env (or MOCK=1 for tests only).` | Set `MODEL_ACCESS_KEY` in `.env` and make sure it is loaded (see next row). There is no automatic mock fallback |
| Key set but startup still fails, or UI shows `MOCK` | `.env` not loaded: with uvicorn, `set -a && . ./.env && set +a`; with compose, `docker compose up -d --force-recreate`. `MOCK` shows only if `MOCK=1` is set somewhere; remove it |
| Evolus shows `not configured` | Set `EVOLUS_API_KEY`, `EVOLUS_APP_ID` and the resource IDs; each block turns `live` once its ID is present |
| `env_file` / `required` error from compose | Upgrade to Docker Compose ≥ 2.24 |
| Diagnosis shows `needs_human` / "model offline" | Check `GET /audit` for the HTTP status (401 = bad key, 404 = wrong `MODEL_NAME` or base URL) |
| Endpoint rejects `response_format` | Handled automatically: the request is retried without it |
| Workflow returns 409 | Turn the Evolus workflow ON |
| UI can't reach the API | `TRUEFIELD_API_URL` (compose sets `http://api:8080`); the API container must be healthy |
| Model retrains on every start | Expected when the dataset changes; the weights are tied to the dataset size and source |
