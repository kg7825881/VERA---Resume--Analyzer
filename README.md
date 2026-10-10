# VERA - Verified Evaluation & Ranking Assistant

VERA - Verified Evaluation & Ranking Assistant is a local-first, evidence-led application for screening a batch of resumes against a job description (JD). It extracts structured information from PDF and DOCX files, applies a repeatable scoring policy, ranks candidates, and shows the evidence behind each outcome.

It supports recruiter decision-making; it does not make autonomous hiring decisions. Recruiters must review inferred evidence, validate missing information, and follow the organisation's approved hiring process.

## Contents

- [Capabilities](#capabilities)
- [Workflow and scoring](#workflow-and-scoring)
- [System Architecture](#system-architecture)
- [Technology stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Local setup](#local-setup)
- [Docker setup](#docker-setup)
- [Using the application](#using-the-application)
- [Configuration](#configuration)
- [Windows desktop installer](#windows-desktop-installer)
- [Quality, privacy, and security](#quality-privacy-and-security)
- [Repository map](#repository-map)

## Capabilities

- Upload PDF/DOCX job descriptions and resume batches.
- Extract role requirements, skills, education, work history, and evidence.
- Stream resume-ingestion results as each document finishes.
- Score candidates across mandatory skills, experience, title fit, soft skills, education, and preferred skills.
- Apply a mandatory-skill coverage gate before designating an automatic match.
- Present evidence as direct, related/inferred, weak, or missing.
- Persist JDs, candidates, runs, results, and feedback in SQLite.
- Run as a browser application, Docker Compose stack, or Electron desktop application.
- Support both controlled local-Ollama judgement and optional browser WebLLM semantic analysis.

## Workflow and scoring

1. A recruiter uploads a JD or selects a saved role.
2. The backend extracts the title, skills, education, experience, and other requirements into a role policy.
3. Resumes are transformed into structured, traceable candidate facts.
4. The matcher first seeks explicit deterministic evidence, including named-skill safeguards.
5. For eligible unresolved requirements, BM25 retrieves relevant resume excerpts.
6. The local judge receives only the requirement and retrieved excerpts, then returns `direct`, `related`, `weak`, or `none`.
7. Python applies fixed weights and the mandatory-coverage gate. Models never calculate or override the final score.
8. Results are stored, ranked, and made available for recruiter review and export.

### Current scoring policy

Weights are configured in `vera-engine/scorer.py` and total 100%.

| Category | Weight | Purpose |
| --- | ---: | --- |
| Mandatory skills | 30% | Subject to the mandatory coverage gate. |
| Relevant experience | 20% | Compares extracted experience against the JD requirement. |
| Job-title match | 10% | Matches recent title evidence against approved title references. |
| Soft skills | 10% | Scores JD soft-skill requirements. |
| Education | 25% | Checks degree and field evidence. |
| Preferred skills | 5% | Scores nice-to-have technical requirements. |

The default mandatory-skill threshold is **60% coverage**. Treat policy and weight changes as controlled product changes: test them against representative, approved benchmark data before release.

## System Architecture

```mermaid
flowchart TB
  Recruiter([Recruiter]) --> Screens

  subgraph Stage1[1. Recruiter Workspace — Next.js / React]
    direction TB
    Screens[Screen Candidates<br/>Ranking · Candidate Detail · Qualified Candidates]
    BrowserState[React Context + localStorage<br/>active role and cached result state]
    Export[Copy summaries<br/>DOCX export · print-ready PDF]
    Screens <--> BrowserState
    Screens --> Export
  end

  Screens <-->|REST requests + NDJSON upload progress| API

  subgraph Stage2[2. Data Ingestion and Normalization — FastAPI / Python]
    direction TB
    API[api.py<br/>upload · analyze · results · feedback]
    Documents[Job descriptions + resumes<br/>PDF / DOCX]
    Extract[Extraction service<br/>text extraction · OCR fallback · source bundles]
    Facts[Normalized facts<br/>JD requirement metadata · candidate profile<br/>skills · education · experience · latest role]
    API --> Documents --> Extract --> Facts
  end

  Facts --> Rules

  subgraph Stage3[3. Evidence Matching and Decision Routing]
    direction TB
    Rules[Deterministic evidence layer<br/>aliases · title families · degree checks · years<br/>explicit named-tool protection]
    Gate{Mode selected?}
    Audit[Deterministic audit<br/>no semantic model call]
    Local[Standard local pipeline<br/>BM25 retrieves only relevant resume chunks]
    BrowserSemantic[Browser Semantic pipeline<br/>semantic retrieval + cross-encoder]
    Ollama[Ollama local judge<br/>structured decision for unresolved evidence]
    WebLLM[WebLLM browser worker<br/>WebGPU judgment for ambiguous pairs only]
    Rules --> Gate
    Gate -->|Audit mode| Audit
    Gate -->|No mode selected| Local
    Gate -->|Browser Semantic| BrowserSemantic
    Local --> Ollama
    BrowserSemantic -->|ambiguous pairs only| WebLLM
  end

  Audit --> Score
  Ollama --> Score
  WebLLM -->|strict JSON validation| API
  API --> Score

  subgraph Stage4[4. Scoring, Persistence, and Recruiter Review]
    direction TB
    Score[Explainable scoring<br/>weighted categories · direct/related/missing<br/>mandatory coverage gate · source citations]
    Store[(SQLite database<br/>JDs · resumes · scores · runs<br/>feedback · candidate-name corrections)]
    Results[Recruiter-ready output<br/>ranking · evidence · summaries · shortlist]
    Score --> Store --> Results
  end

  Results --> Screens

  classDef ui fill:#17304a,stroke:#55d6c2,color:#eef6ff;
  classDef ingestion fill:#102238,stroke:#7c8cff,color:#eef6ff;
  classDef decision fill:#3a2e18,stroke:#f5bd62,color:#eef6ff;
  classDef data fill:#14332f,stroke:#57d38c,color:#eef6ff;
  class Screens,BrowserState,Export ui;
  class API,Documents,Extract,Facts ingestion;
  class Rules,Gate,Audit,Local,BrowserSemantic,Ollama,WebLLM decision;
  class Score,Store,Results data;
```


## The three operating modes

| UI selection | What happens | Model usage |
| --- | --- | --- |
| No mode selected | The standard local pipeline runs. Exact rules are attempted first; unresolved evidence can be judged by local Ollama. | Ollama may be called for non-exact evidence. |
| **Deterministic audit mode** | Uses source-backed parsing, aliases, title/education/experience rules, and deterministic matching only. | No semantic judge call. |
| **Browser semantic cascade** | The API performs deterministic checks, retrieval, and cross-encoder routing. Only remaining ambiguous candidate-requirement pairs are sent to the browser for a constrained WebLLM judgment. | WebLLM only for ambiguity; no Ollama semantic judging for those pairs. |

Browser semantic mode is intentionally not a broad “ask an LLM about the whole JD” operation. The backend creates one candidate-isolated semantic session, gives WebLLM a compact ambiguity batch, validates its structured response, and only then finalizes and persists the score.


### Trust boundaries

- Deterministic code owns data validation, requirements policy, arithmetic, mandatory gates, and final ranking.
- Model assistance is limited to classifying retrieved evidence; it is not permitted to invent evidence or calculate scores.
- Invalid/unavailable model output fails closed rather than becoming a positive match.

## Technology stack

| Area | Technologies | Source location |
| --- | --- | --- |
| Web UI | Next.js 14, React 18, App Router | `vera-frontend/app`, `vera-frontend/components` |
| State/API client | React Context, Fetch API | `app/providers.js`, `lib/api.js` |
| Browser semantic option | `@mlc-ai/web-llm`, Web Workers | `lib/webllm.js`, `workers/webllm-worker.js` |
| Backend API | FastAPI, Uvicorn, Pydantic | `vera-engine/api.py` |
| Document extraction | PyMuPDF, PyMuPDF4LLM, pdfplumber, python-docx, docx2txt | `vera-engine/extractor.py` |
| OCR fallback | Tesseract, pdf2image, Poppler | `extractor.py`, `vera-engine/Dockerfile` |
| Retrieval and judgement | rank-bm25, Ollama | `retrieval.py`, `judge.py` |
| Deterministic policy | Python scoring/matching rules | `scorer.py`, `matcher.py`, `decision_scoring.py` |
| Persistence | SQLite | `vera-engine/db.py` |
| Containers | Docker, Docker Compose | `Dockerfile`s, `compose.yaml` |
| Desktop distribution | Electron, electron-builder, PyInstaller | `vera-desktop`, `vera-engine/vera-api.spec` |
| Tests | pytest | `vera-engine/test_*.py` |

## Prerequisites

For local development, install:

| Requirement | Purpose |
| --- | --- |
| Python 3.10+ | Backend runtime |
| Node.js 20 LTS + npm | Frontend and desktop packaging |
| Ollama | Local evidence-judge route |
| Tesseract OCR + Poppler | OCR/image-only PDF fallback |

Pull the models configured by the current codebase:

```bash
ollama pull qwen2.5:3b-instruct
ollama pull gemma3:4b
```

On Windows, install Tesseract and Poppler and ensure their executable folders are on `PATH`. Verify them in a new PowerShell window:

```powershell
tesseract --version
pdftoppm -v
```

The optional Browser Semantic mode also requires a WebGPU-capable browser/device and sufficient memory to load its browser model.

## Local setup

Use two terminals from the repository root.

### Backend

**macOS/Linux**

```bash
cd vera-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

**Windows PowerShell**

```powershell
cd vera-engine
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API runs at `http://localhost:8000`; its OpenAPI document is at `http://localhost:8000/openapi.json`.

### Frontend

In a second terminal:

```bash
cd vera-frontend
npm ci
npm run dev
```

Open `http://localhost:3000/screen`.

To point the frontend at a different backend, create `vera-frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

`NEXT_PUBLIC_*` variables are compiled into the browser bundle, so rebuild after changing them.

## Docker setup

Docker Compose starts both services and stores the database/uploaded documents in named volumes.

```bash
docker compose up --build -d
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

Stop services without deleting data:

```bash
docker compose down
```

Reset Docker-managed data only when intended:

```bash
docker compose down -v
```

The Docker backend expects Ollama at `http://host.docker.internal:11434`. Ensure the host Ollama service is running and accessible to Docker.

## Using the application

1. Open **Screen Candidates**.
2. Upload a JD or select one from the library.
3. Review the extracted role summary and upload PDF/DOCX resumes.
4. Optionally correct candidate display names before analysis.
5. Choose a mode:
   - **Default/local route:** deterministic matching plus controlled local-Ollama evidence judgement where needed.
   - **Deterministic audit mode:** does not call the semantic judge.
   - **Browser semantic cascade:** optional WebLLM processing where supported.
6. Select **Run live analysis**.
7. Review **Candidate Ranking**, open candidate details, and inspect the supporting evidence.
8. Use **Qualified Candidates** to review/export shortlist results.

## Configuration

Configure these values in the backend environment or in the frontend's `.env.local` file. The defaults work for local development.

### Frontend

- **`NEXT_PUBLIC_API_BASE_URL`**  
  Browser-visible FastAPI base URL. Default: `http://localhost:8000`.

### Backend runtime

- **`TALENTLENS_FRONTEND_ORIGINS`**  
  Comma-separated CORS allowlist. Default: `http://localhost:3000`.

- **`TALENTLENS_DB_PATH`**  
  SQLite database location. Default: `VERA.db`.

- **`TALENTLENS_MAX_WORKERS`**  
  Maximum number of concurrent API-side processing workers. Default: `4`. Set this according to available CPU, RAM, and Ollama capacity.

### AI and semantic services

- **`OLLAMA_HOST`**  
  Address of the local or remote Ollama service. Default: the Ollama client default.

- **`TALENTLENS_EMBED_MODEL`**  
  Embedding model configuration. Default: `nomic-embed-text`.

- **`TALENTLENS_CROSS_ENCODER_MODEL`**  
  Cross-encoder configuration. Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Example backend configuration:

```dotenv
TALENTLENS_FRONTEND_ORIGINS=http://localhost:3000
TALENTLENS_DB_PATH=VERA.db
TALENTLENS_MAX_WORKERS=4
OLLAMA_HOST=http://127.0.0.1:11434
```

## Windows desktop installer

The Electron package launches the backend and standalone frontend locally, then displays the frontend in a desktop window.

### Runtime checks before the app window opens

The desktop shell verifies that:

1. the bundled backend executable exists;
2. the bundled frontend `server.js` exists;
3. the backend responds at `http://127.0.0.1:8000/openapi.json`; and
4. the frontend responds at `http://127.0.0.1:3000`.

If a check fails, the application shows a startup error rather than opening a partial UI. The app also requires 64-bit Windows, permission to write per-user application data, and local ports 8000/3000 to be available. Ollama remains an additional requirement for the local semantic-judge route.

### Build prerequisites

- Windows 10/11 (64-bit)
- Node.js and npm
- Python 3.10+, backend dependencies, and PyInstaller
- The Windows Electron packaging workspace (`vera-desktop_windows` in the Windows distribution checkout)

### Build procedure

1. Build the backend executable on Windows:

   ```powershell
   cd vera-engine
   py -m pip install -r requirements.txt pyinstaller
   py -m PyInstaller vera-api.spec
   ```

2. Build the standalone frontend:

   ```powershell
   cd ..\vera-frontend
   npm ci
   npm run build
   ```

3. Copy static assets into the package runtime. For example, verify the logo exists before distribution:

   ```powershell
   Test-Path .\dist\win-unpacked\resources\frontend\public\gigaforce-logo.png
   ```

4. Package the Windows installer:

   ```powershell
   cd ..\vera-desktop_windows
   npm ci
   npm run dist
   ```

5. Find the generated installer in `vera-desktop_windows\dist\`, install it, and smoke-test that fresh build.

> The checked-in `vera-desktop` package contains the shared Electron runtime and Apple Silicon packaging script. Keep the Windows workspace's `extraResources` aligned with the backend bundle, frontend runtime, dependencies, and `public` assets.

### macOS package

```bash
cd vera-frontend && npm ci && npm run build
cd ../vera-desktop && npm ci && npm run dist:mac
```

The backend bundle must exist at the path configured in `vera-desktop/package.json`.

## Quality, privacy, and security

### Release checks

```bash
cd vera-engine && pytest
cd ../vera-frontend && npm run build
```

Before release, test a PDF and DOCX upload, known scoring/evidence outcomes, audit and semantic modes, a clean desktop install, database persistence, and static assets such as the GigaForce logo.

### Data handling

- Resumes and JDs contain sensitive personal data. Protect repositories, database backups, Docker volumes, and desktop user-data folders according to company policy.
- The application is an internal/local tool and does not provide multi-user authentication or authorisation. Do not expose it publicly without authentication, HTTPS, secret management, logging, access controls, and a security review.
- Never store credentials in `NEXT_PUBLIC_*` variables.
- Uploaded files are retained locally by the backend. Remove them deliberately according to retention policy.
- Do not attach resumes or candidate data to public issue trackers.

## Repository map

```text
.
├── compose.yaml                 # Docker Compose stack
├── vera-engine/                 # FastAPI, extraction, matching, scoring, SQLite
│   ├── api.py                   # HTTP endpoints/startup
│   ├── extractor.py             # PDF/DOCX extraction and OCR fallback
│   ├── matcher.py               # Evidence matching policy
│   ├── retrieval.py             # BM25 retrieval
│   ├── judge.py                 # Controlled Ollama evidence judgement
│   ├── scorer.py                # Deterministic scoring and coverage gate
│   └── test_*.py                # Backend tests
├── vera-frontend/               # Next.js recruiter interface
│   ├── app/                     # Routes, state, layout, CSS
│   ├── components/              # UI components
│   ├── lib/                     # API client/adapters/semantic logic
│   ├── workers/                 # WebLLM worker
│   └── public/                  # Static assets
└── vera-desktop/                # Electron runtime and packaging support
    ├── main.cjs                 # Starts local services and Electron window
    └── scripts/                 # Prepares standalone frontend runtime
```

## Support

When raising an internal support request, include the application version, operating system, Node/Python versions, selected analysis mode, and the relevant startup/API error. Do not include candidate documents or personally identifiable information in public channels.
