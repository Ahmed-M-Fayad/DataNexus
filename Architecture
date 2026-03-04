# DataNexus — Architecture Decisions

This document records the major structural decisions made during Step 3 of system analysis.
Every decision includes what was chosen and why, so future contributors understand the reasoning.

---

## Diagrams

All system design files live under `docs/system-design/`.

| Format | File                                            |
|--------|-------------------------------------------------|
| SVG    | `docs/system-design/datanexus_architecture.svg` |

---

## System Layers

The system is organized into 4 layers. Each layer communicates only with the layer directly below it — no layer skips another.

| Layer | Name              | Components                                      |
|-------|-------------------|-------------------------------------------------|
| 1     | User Interaction  | Streamlit dashboard, CLI (Click), Flask REST API|
| 2     | Orchestration     | Apache Airflow (validation DAG)                 |
| 3     | Core Engine       | Data Profiler, Validation Engine, Alert Manager, GE Adapter |
| 4     | Storage           | PostgreSQL (single source of truth)             |

---

## Decisions

### 1. Asynchronous Validation Execution
**Decision:** When the Flask API receives `POST /api/validations/:id/run`, it triggers an Airflow DAG and returns `202 Accepted` immediately. It does not wait for the run to finish.

**Why:** Validation runs on large datasets can take minutes. A synchronous approach would time out HTTP connections and give poor user experience. The client polls `GET /api/runs/:id` to check status.

---

### 2. Validation Config Stored in PostgreSQL
**Decision:** YAML validation configs are stored as text in a `pipeline_configs` database table, not as files on disk.

**Why:** Storing configs in the database allows the Streamlit UI to create and edit them without touching the filesystem, and allows the API to serve them without filesystem access. It also means configs are versioned alongside results.

---

### 3. Great Expectations Wrapped Behind an Adapter
**Decision:** The rest of the system never imports Great Expectations directly. All GE interaction goes through `src/ge_adapter/ge_adapter.py`.

**Why:** This is the Adapter Pattern. If we ever need to swap GE for another library (Pandera, Soda Core), we only change one file. No other component needs to know GE exists.

---

### 4. Docker Compose for Deployment
**Decision:** The project runs via Docker Compose with 4 containers: `postgres`, `airflow-webserver`, `airflow-scheduler`, `api`, and `dashboard`.

**Why:** Docker Compose is the right complexity level for a team project. It is reproducible across all team members' machines and does not require Kubernetes-level infrastructure knowledge.

---

### 5. PostgreSQL as the Single Source of Truth
**Decision:** All persistent data — run results, quality scores, configs, alerts, profiles — lives only in PostgreSQL. No component writes permanent data to disk or memory.

**Why:** Having one data store eliminates consistency bugs. If Streamlit reads data and the CLI reads data, they both see the same thing because they both query the same database.

---

### 6. Secrets via Environment Variables
**Decision:** All credentials (database password, Airflow key, Slack webhook, SMTP password) are loaded from a `.env` file using `python-dotenv`. A `.env.example` file is committed to the repo as a template.

**Why:** Follows the Twelve-Factor App methodology. Credentials are never hardcoded or committed to Git.

---

### 7. API Authentication via API Keys
**Decision:** The Flask API authenticates requests using an `X-API-Key` header, validated in a Flask middleware/decorator. There is no OAuth or JWT at this stage.

**Why:** Simple and sufficient for the project scope. The API is called by Airflow and GitHub Actions, not end users, so API keys are appropriate.