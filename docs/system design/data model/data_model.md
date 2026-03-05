# DataNexus — Data Model Decisions
This document records the major structural decisions made during Step 3 of system analysis.
Every decision covers what was chosen and why, so future contributors understand the reasoning behind the schema.

---

## Diagrams

All system design files live under `docs/system-design/`.

| Format | File                                              |
|--------|---------------------------------------------------|
| PDF    | `docs/system-design/datanexus_datamodel.pdf`      |

---

## Entity Overview

The schema is organized into 8 tables grouped by concern.

| Group               | Tables                                                        |
|---------------------|---------------------------------------------------------------|
| Ingestion           | `DATA_SOURCES`, `DATASETS`                                    |
| Profiling           | `DATA_PROFILES`                                               |
| Validation Config   | `VALIDATION_CONFIGS`, `TEST_DEFINITIONS`                      |
| Validation Execution| `VALIDATION_RUNS`, `VALIDATION_RESULTS`                       |
| Observability       | `PIPELINE_METRICS`, `ALERTS`                                  |

---

## Relationships

| From                | Relationship    | To                  | Cardinality |
|---------------------|-----------------|---------------------|-------------|
| `DATA_SOURCES`      | feeds           | `DATASETS`          | 1 : N       |
| `DATASETS`          | profiled via    | `DATA_PROFILES`     | 1 : N       |
| `DATASETS`          | defines rules for| `VALIDATION_CONFIGS`| 1 : N       |
| `DATASETS`          | logs metrics to | `PIPELINE_METRICS`  | 1 : N       |
| `VALIDATION_CONFIGS`| executes        | `VALIDATION_RUNS`   | 1 : N       |
| `VALIDATION_CONFIGS`| references      | `TEST_DEFINITIONS`  | N : N (via config) |
| `VALIDATION_RUNS`   | produces        | `VALIDATION_RESULTS`| 1 : N       |
| `VALIDATION_RUNS`   | triggers        | `ALERTS`            | 1 : N       |

---

## Decisions

### 1. `DATA_SOURCES` Holds Connection Strings, Not Credentials

**Decision:** `DATA_SOURCES` stores a `Connection_string` field as a plain string alongside a `Status` and `Last_tested_at` timestamp. Credentials embedded in the connection string are injected at runtime from environment variables, not stored in plaintext in the database.

**Why:** The table needs to know *where* a source lives (host, port, database name) so it can be referenced by multiple datasets. Separating the concept of "where" from "credentials" means we can store connection metadata safely. The `Last_tested_at` field allows the UI to show whether a source is reachable without requiring a live connection test on every page load.

---

### 2. `DATASETS` Decoupled from `DATA_SOURCES`

**Decision:** `DATASETS` is its own table with a `Source_ID` foreign key back to `DATA_SOURCES`. A dataset stores `Schema_name` and `Table_name` separately from the source's connection details.

**Why:** One data source (e.g., a single PostgreSQL instance) can contain many datasets (e.g., `public.customers`, `public.orders`, `finance.invoices`). Decoupling them means adding a new dataset from an existing source requires no duplication of connection metadata. It also lets us attach profiling configs, validation configs, and metrics independently to each dataset.

---

### 3. `DATA_PROFILES` Stores a Full Snapshot in `Profile_json`

**Decision:** `DATA_PROFILES` stores scalar summary fields (`Row_count`, `Column_count`, `Data_size`) alongside a `Profile_json` column that holds the full column-level statistical profile as a JSON document.

**Why:** Scalar fields allow fast SQL queries like "which datasets grew more than 20% since yesterday?" without deserializing JSON. The `Profile_json` blob holds the rich per-column detail (null rates, distributions, detected patterns) that the dashboard and profiler engine need. Splitting them avoids wide tables with hundreds of nullable columns while still keeping everything in one row per profiling run.

---

### 4. `VALIDATION_CONFIGS` Owns the Schedule and Alert Channel List

**Decision:** `VALIDATION_CONFIGS` stores `Schedule_cron` directly on the config row rather than in a separate scheduling table.

**Why:** Each validation config has exactly one schedule. A separate scheduling table would add a join for no benefit at this scale. Storing the cron string directly lets Airflow's DAG generator read one row and produce a fully-configured DAG without cross-table lookups. If a config is deactivated, the DAG is simply not triggered — no orphan schedule records are left behind.

---

### 5. `TEST_DEFINITIONS` as a Reusable Library

**Decision:** `TEST_DEFINITIONS` is a standalone table that stores reusable, named test templates (`Name`, `Category`, `Description`, `Implementation_code`). `VALIDATION_CONFIGS` references definitions rather than duplicating logic inline.

**Why:** Without a library, the same "not null" or "email regex" check would be copy-pasted into every config. A library means a bug fix in `Implementation_code` propagates to every config that references it. The `Category` field (`completeness`, `accuracy`, `consistency`, `timeliness`, `uniqueness`) also lets the dashboard group and filter checks by type without parsing implementation code.

---

### 6. `VALIDATION_RUNS` Captures Both Schedule and Trigger Source

**Decision:** `VALIDATION_RUNS` stores `Run_date`, `Schedule_due`, `Triggered_by`, and `Is_active` on the same row.

**Why:** A run can be started three ways: on schedule (Airflow), manually (CLI or API), or via CI/CD (GitHub Actions). Storing `Triggered_by` on the run record lets us distinguish scheduled quality regressions from manual re-runs in historical reports. `Schedule_due` captures *when* the run was supposed to start, so SLA latency (actual start vs. scheduled start) can be computed without joining to the config's cron expression.

---

### 7. `VALIDATION_RESULTS` Stores `Expected_value` and `Actual_value` as Strings

**Decision:** `VALIDATION_RESULTS` stores `Expected_value` and `Actual_value` as string columns rather than typed numeric fields.

**Why:** Different check types produce fundamentally different result shapes. A null-check result might be `"95% non-null"`. A regex result might be `"pattern: ^[a-z]+@.*"`. A range result might be `"[18, 120]"`. Using a single string column for each accommodates all check types without nullable typed columns for every possible result format. The `Profile_json` column on the same row holds the structured breakdown for consumers that need to parse it.

---

### 8. `ALERTS` Referenced by `Run_ID`, Not `Dataset_ID`

**Decision:** `ALERTS` holds a foreign key to `VALIDATION_RUNS`, not directly to `DATASETS` or `VALIDATION_CONFIGS`.

**Why:** An alert is the result of a specific run, not of a dataset in the abstract. Anchoring to the run means the alert record carries a precise point-in-time reference. From `Run_ID` you can always join up to `Config_ID` and then to `Dataset_ID` when needed. Going the other direction — storing only `Dataset_ID` on the alert — would lose the exact run context and make deduplication harder.

---

### 9. `PIPELINE_METRICS` as a Separate Aggregation Table

**Decision:** `PIPELINE_METRICS` is its own table storing one row per dataset per day, with pre-aggregated fields: `Records_processed`, `Processing_time_sec`, `Avg_quality_score`, `Total_validation_failures`, `Sla_met`.

**Why:** These metrics *could* be computed on the fly by aggregating `VALIDATION_RUNS` and `VALIDATION_RESULTS`. In practice, that query joins four tables and scans potentially millions of result rows every time the dashboard loads. Writing one summary row per day at the end of each run makes dashboard queries a simple `SELECT` against a narrow, indexed table. The `Metadata` JSON column preserves any additional context without requiring schema migrations for new metric types.

---

### 10. Timestamps Named `Created_at` Consistently Across All Tables

**Decision:** Every table uses `Created_at` (and `Updated_at` where rows are mutable) with a database-level default of `CURRENT_TIMESTAMP`. No table uses `inserted_on`, `date_added`, or other variations.

**Why:** Consistent naming means any developer can predict the audit column name without checking the schema. Database-level defaults ensure the timestamp is always set even if application code forgets to include it. `Updated_at` is maintained by a trigger (not application code) so it cannot be accidentally omitted on update.

---

## Column Naming Conventions

| Convention               | Example                                  |
|--------------------------|------------------------------------------|
| Primary keys             | `ID` (integer, all tables)               |
| Foreign keys             | `<Table>_ID` (e.g., `Dataset_ID`)        |
| Booleans                 | `Is_<state>` or past-tense verb (e.g., `Is_active`, `Acknowledged`, `Sla_met`) |
| Timestamps               | `<Event>_at` (e.g., `Created_at`, `Sent_at`, `Last_tested_at`) |
| Free-form structured data| `<Context>_json` (e.g., `Profile_json`, `Metadata`) |

---

## What Is Not in the Schema (and Why)

| Omitted Concept         | Reason                                                                                  |
|-------------------------|-----------------------------------------------------------------------------------------|
| User / auth tables      | Auth is handled at the API layer via API keys (see Architecture Decisions §7). No per-user data is stored in this version. |
| Data lineage edges      | Lineage tracking is a planned Phase 2 feature. Adding it now would require a graph structure that is out of scope for the MVP. |
| Soft-delete flags       | Rows are hard-deleted or cascade-deleted via foreign key constraints. Soft deletes add query complexity with no benefit at this scale. |
| Separate scheduling table | Schedule is stored directly on `VALIDATION_CONFIGS` (see Decision §4 above). |

---

*Document Version: 1.0*
*Last Updated: March 2026*
*Prepared for: DataNexus / DEPI Graduation Project*
