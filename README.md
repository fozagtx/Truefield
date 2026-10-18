# Truefield: Maintenance Copilot for Factory Technicians

*Truefield turns a vibration spike into a safe, signed-off work order in under two minutes.*

AMD Developer Hackathon ACT III · Track 1 Intelligent Industry · Evolus Business Agents · Google Gemma

Truefield works in four steps:

1. **Detect.** A PyTorch autoencoder (ROCm on AMD MI300X, CPU fallback) plus deterministic physics features (RMS, kurtosis, BPFO envelope energy) flags a failing bearing early, on replayed **NASA IMS Set 2** run-to-failure data.
2. **Explain.** An LLM (default `gpt-4o` through an OpenAI-compatible inference API) explains the likely cause and cites the manual and past work orders. Output is strict JSON, validated against a schema. Optional alternatives: the Evolus agent *Truefield Diagnostician*, or our own vLLM endpoint (e.g. Gemma on AMD).
3. **Draft and gate.** An Evolus workflow drafts a work order. Lockout/tagout (LOTO) steps always come from a **deterministic rules engine** and the LLM can't remove them. High-risk or low-confidence cases need **supervisor approval**.
4. **Close out.** The technician closes the job with a photo, PDF or voice note (Evolus extractor / AI Tool `minutes`). The result becomes structured asset history and is fed back into the knowledge library.

> **Data honesty.** The vibration data is real NASA IMS data once you run the download script. The housing **temperature channel is synthetic** (derived from the vibration RMS) and is labelled that way everywhere. With no dataset downloaded, the app uses a small **synthetic vibration fallback**, and the UI badge says so. The 12 historic work orders and the plant LOTO procedure in `kb/` are synthetic demo documents.

---

## Quickstart (mock mode, no keys, about 2 minutes)

```bash
git clone https://github.com/fozagtx/Truefield && cd Truefield
docker compose up --build
# UI  -> http://localhost:8501     API -> http://localhost:8080/docs
```

With no `.env`, every external call runs in **mock mode**: the diagnosis model, the Evolus workflow, extractor, AI Tools and library. Mock calls are still written to the audit log and labelled `mock` in the UI. On first start the API trains the autoencoder on the CPU (a few seconds) and saves `models/ae-v1.pt`.

**Without Docker:**

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
uvicorn truefield.api.main:app --port 8080 &
TRUEFIELD_API_URL=http://localhost:8080 streamlit run ui/app.py
pytest -q                       # 44 tests: rules (100% branch coverage), features, schemas, LLM layer, Evolus client, end-to-end API
```

---

## Architecture

```mermaid
flowchart LR
  subgraph AMD["AMD Developer Cloud: MI300X droplet (ROCm)"]
    VLLM["vLLM: Gemma 4 26B-A4B-it<br/>OpenAI-compatible :8000 /v1<br/>--api-key"]
    AE["PyTorch on ROCm<br/>Autoencoder train + score"]
  end
  subgraph APP["Truefield app (Docker Compose)"]
    REPLAY["Telemetry replay<br/>NASA IMS Set 2 + synthetic temperature"]
    FEAT["Feature extractor<br/>RMS / kurtosis / BPFO envelope"]
    RULES["Rules engine<br/>(deterministic)"]
    API["FastAPI orchestrator"]
    DB[("SQLite<br/>events, WOs, history, audit")]
    UI["Streamlit UI<br/>Technician / Supervisor / Manager"]
  end
  subgraph EVOLUS["Evolus event workspace"]
    AGENT["Agent: Truefield Diagnostician<br/>(model = our vLLM Gemma)"]
    LIB["Knowledge library<br/>manuals, LOTO, past WOs"]
    WF["Workflow: Draft Work Order"]
    EXT["Extractor: Close-out form"]
    TOOLS["AI Tools: minutes (run-media)"]
  end
  REPLAY --> FEAT --> AE --> RULES --> API
  API -- "/v1/chat/completions" --> AGENT
  AGENT --- LIB
  AGENT -- "model calls" --> VLLM
  API -- "POST /workflows/{id}/execute" --> WF
  WF -- "agent step" --> AGENT
  WF -- "callbackUrl / polling" --> API
  API -- "ask-file" --> EXT
  API -- "ai-tools/minutes/run-media" --> TOOLS
  API -- "upload closed WO" --> LIB
  API <--> DB
  UI <--> API
```

| Layer | Module |
|---|---|
| Features (FR-2) | `truefield/anomaly/features.py`: Hilbert envelope of the 2–8 kHz band; energy within ±3 Hz of BPFO 236 / BPFI 297 / BSF 278 / FTF 15 Hz |
| Anomaly model (FR-3) | `truefield/anomaly/{model,train,score}.py`: AE 32→16→4→16→32 on log-FFT bands, trained on the first 40% (healthy); threshold is the p99.5 of the healthy error; 3-snapshot debounce. Device auto-detect: `cuda` (ROCm/HIP or CUDA), else CPU |
| Rules (FR-4) | `truefield/rules/{rules.yaml,engine.py}`: R-SEV-1..4, R-HYP-1..3, R-RISK, R-ROUTE, R-SAFE-1..3, R-PRI |
| LLM (FR-5) | `truefield/llm/{client,diagnose}.py`: OpenAI-compatible; primary = `MODEL_BASE_URL` with `MODEL_ACCESS_KEY` (system prompt sent as role `developer`; `response_format` JSON mode, dropped automatically if the endpoint rejects it); optional Evolus agent or vLLM; schema validation, one repair retry, then `needs_human`; citation docs checked against the library |
| Evolus (FR-6/8/9) | `truefield/evolus/{client,agent,library,workflow,extractor,tools}.py`: timeout + one retry, audit log, mock mode with no key |
| Orchestration | `truefield/pipeline.py`, `truefield/api/main.py`, `truefield/store/db.py`, `truefield/kpi.py` |
| UI | `ui/app.py` (Streamlit) |

**API:** `POST /replay/start?speed=&start_frac=` · `GET /telemetry` · `GET /events` · `POST /events/{id}/diagnose` · `POST /events/{id}/workorder` · `POST /wo/{id}/approve` · `POST /wo/{id}/issue` · `POST /wo/{id}/closeout` (multipart: `source`, `file`, `overrides`) · `POST /callback/evolus` · `GET /kpi` · `GET /bench` · `GET /audit` · `GET /assets/PUMP-07/BRG-1` · `POST /reset`. Interactive docs are at `/docs`.

---

## Safety design

- **Deterministic core, narrow LLM.** Severity, fault hypothesis, risk, routing and LOTO come from `rules.yaml` and `engine.py`: pure Python, 100% branch-covered tests, same input → same output.
- **The LLM can only escalate.** `final.severity = max(rules, llm)`; the same applies to risk. Downgrade attempts are logged as `guardrail_triggered` and counted on the KPI panel.
- **LOTO is always present.** The rules' LOTO entries are merged into every work order, whatever the workflow or LLM returns. Extra LLM LOTO entries are kept only if cited. A work order can't reach `issued` without LOTO.
- **Human gates.** Supervisor approval is required for high risk, confidence < 0.6, a status other than `diagnosed`, LLM disagreement with the rules, or any uncited step (R-ROUTE-1). Technicians can't issue a `pending_approval` work order (HTTP 403). Approve, edit and reject are logged with the user and a timestamp.
- **Grounding.** Diagnoses must cite at least one library document. Citations to unknown documents are dropped, and a diagnosis left with none goes to `needs_human`. Parts not found in the diagnosis are dropped from work orders.
- **Close-out integrity.** `technician`, `root_cause_confirmed`, `actions_done` and `loto_removed_by` are required. The voice parser **never guesses** who removed the locks.
- **Failure handling.** Every external call has a timeout and one retry. If the model is offline, the case goes to the supervisor with a rules-only summary. Workflow 409 or failure → direct agent call with the §13.2 prompt → deterministic draft. With no callback, the app polls every 3 s. `DEMO_CACHE=1` replays recorded responses, labelled *cached*.
- **Never actuates equipment.** Every work order states: *"Draft generated by Truefield. Authorised person must verify isolation before work."*
- **Secrets** live only in env vars (`.env` is git-ignored). No secrets go into workflow parameters, because Evolus stores parameters with the run.

---

## Setup

### 1. Primary LLM (one key)

```bash
cp .env.example .env      # set MODEL_ACCESS_KEY (MODEL_BASE_URL / MODEL_NAME default to https://inference.do-ai.run/v1 and gpt-4o)
docker compose up --build
```

With `MODEL_ACCESS_KEY` set, diagnosis and work-order drafting use the live model. The UI badge shows the model name. Evolus blocks stay in mock mode until you configure them.

### 2. Evolus event workspace (optional)

1. Create the workspace **from the event link** (evolus.ai/en/hackathon), then create an API key. The API keys page shows the space ID (`EVOLUS_APP_ID`).
2. **Knowledge library** `truefield-kb` (language `en`): set `EVOLUS_LIBRARY_ID` and run `python scripts/seed_library.py`.
3. **Agent** `Truefield Diagnostician`: attach `truefield-kb`, use the system prompt in `truefield/prompts/diagnostician.txt`, temperature 0.2. Set `EVOLUS_AGENT_ID`; the chat model id `<appId>:<agentId>` is derived from it.
4. **Workflow** `Draft Work Order`: API trigger; parameters `asset_id` and `route`; an agent step with `truefield/prompts/draft_wo.txt`; a webhook step. **Turn it on** (otherwise you get a 409). Set `EVOLUS_WORKFLOW_ID`. The app polls for the result. For callbacks, set `PUBLIC_BASE_URL` and `TRUEFIELD_CALLBACK_KEY`.
5. **Extractor** `Close-out form`: JSON output with the CloseOut fields (`truefield/schemas/closeout.json`). Set `EVOLUS_EXTRACTOR_ID`; it is assumed to belong to `EVOLUS_AGENT_ID`.

When `MODEL_ACCESS_KEY` is set, the inference API answers diagnosis calls. Set `LLM_PROVIDER=evolus` to route them through the Evolus agent instead.

### 3. AMD GPU (optional): real data, ROCm training, own vLLM

```bash
DATA_DIR=./data ./scripts/download_ims.sh          # 1.08 GB NASA IMS -> data/ims/2nd_test (needs p7zip-full / unrar)
python scripts/make_demo_slice.py --stride 2       # data/demo/ims_set2_b1.npz (real data, not committed)
./deploy/train_rocm.sh                             # trains models/ae-v1.pt on the MI300X and runs bench/ae_bench.py
HF_TOKEN=hf_xxx ./deploy/serve_gemma.sh            # optional vLLM Gemma endpoint; use with LLM_PROVIDER=vllm VLLM_BASE_URL=... VLLM_API_KEY=...
python bench/collect_results.py                    # -> bench/results.json (Bench tab)
```

### Environment variables

`.env.example` lists only the essentials:

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_ACCESS_KEY` | Bearer key for the primary LLM API; empty → mock LLM | |
| `MODEL_BASE_URL` | OpenAI-compatible base URL | `https://inference.do-ai.run/v1` |
| `MODEL_NAME` | model | `gpt-4o` |
| `EVOLUS_API_KEY` | Evolus key; empty → all Evolus calls run in mock mode | |
| `EVOLUS_APP_ID` | space ID → `X-Application-Id` | |
| `EVOLUS_AGENT_ID` | Truefield Diagnostician agent (chat model `<appId>:<agentId>` and extractor owner are derived) | |
| `EVOLUS_WORKFLOW_ID` | Draft Work Order workflow | |
| `EVOLUS_EXTRACTOR_ID` | close-out extractor | |
| `EVOLUS_LIBRARY_ID` | knowledge library | |

Everything else has a code default in `truefield/config.py` and can still be overridden with an env var. Examples: `LLM_PROVIDER` (`auto`/`do`/`evolus`/`vllm`/`mock`), `VLLM_BASE_URL`, `VLLM_API_KEY`, `PUBLIC_BASE_URL`, `DEMO_CACHE`, `COST_PER_HOUR_USD`, `TRUEFIELD_DATASET`.

### Evolus endpoints still to confirm

Every unverified endpoint shape is isolated in `truefield/evolus/*.py` and marked `TODO: confirm in workspace API reference`:

- the workflow execute **parameters** form field name (`parameters` assumed), and the field that holds the final output in an execution or callback;
- which agent ID owns the extractor, and the `ask-file` response envelope;
- the `import-url` body field and the list-documents envelope;
- AI Tool `run` body field names, and slugs other than `minutes` / `translate` / `anonymize`.

---

## Benchmarks

`bench/results.json` is rendered on the Bench tab and at `GET /bench`. The numbers committed now are a **CPU-only run from the dev box, in mock mode**. Replace them with MI300X numbers from `deploy/train_rocm.sh`, `bench/vllm_bench.sh` and `bench/e2e_latency.py` against the live stack. The autoencoder is deliberately tiny, so CPU inference is already sub-millisecond. The GPU's main job on AMD is serving Gemma 4 and training at scale.

Lead time: `python scripts/eval_leadtime.py` replays the whole record and compares Truefield's first alert with a naive 2× healthy-RMS alarm. On the synthetic fallback, Truefield alerts 15.8 h before the end of the record, against 8.3 h for the naive alarm. **Run it on the real IMS data and quote those numbers instead.**

## Repo layout

```
truefield/  api/ anomaly/ rules/ evolus/ llm/ schemas/ store/ data/ prompts/ pipeline.py kpi.py config.py
ui/app.py            Streamlit UI          kb/        library documents (synthetic plant docs, labelled)
scripts/             download_ims.sh, make_demo_slice.py, seed_library.py, eval_leadtime.py, record_demo_cache.py, make_samples.py
bench/               vllm_bench.sh, ae_bench.py, e2e_latency.py, collect_results.py, results.json
deploy/              serve_gemma.sh, docker-compose.vllm.yml, train_rocm.sh
samples/             close-out form (PDF/JPG/TXT), voice-note transcript
tests/               pytest suite
```

## Data citation

J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007), IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center. The data is downloaded by the script and **not redistributed** in this repo.

## License

MIT. See `LICENSE`.
