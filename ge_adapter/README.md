# GE Adapter — Interface Documentation

This module wraps Great Expectations behind a single interface.
**No other file in the project should import `great_expectations` directly.**
All GE functionality goes through `GEAdapter` only.

---

## Quick Start
```python
from src.ge_adapter import GEAdapter
import pandas as pd

adapter = GEAdapter()
result  = adapter.run_expectation(df, check)
```

---

## Input 1 — DataFrame

A standard pandas DataFrame containing the dataset to validate.
```python
df = pd.DataFrame({
    "email": ["user@example.com", "admin@test.com", None],
    "age":   [25, 17, 30],
})
```

---

## Input 2 — Check Dict

A dictionary describing the check to run. Confirmed with the
Validation Engine team. Shape:
```python
check = {
    "name":       "age_in_range",   # str   — check identifier (required)
    "check_type": "range",          # str   — see supported types below (required)
    "column":     "age",            # str   — column name to check (required)
    "threshold":  0.95,             # float — pass threshold 0.0–1.0 (required)
                                    #         e.g. 0.95 = 95% of rows must pass
                                    #         this is passed to GE as 'mostly'
    "severity":   "high",           # str   — low/medium/high/critical (required)

    # Optional — include only what your check type needs:
    "min":        18,               # used by: range, freshness
    "max":        120,              # used by: range, freshness
    "regex":      "^[^@]+@[^@]+$", # used by: regex only
                                    # do NOT supply regex for not_empty —
                                    # it is auto-injected by the adapter
    "values":     ["active", "inactive"], # used by: in_set, foreign_key,
                                          #          referential_integrity
}
```

> **Required keys:** `name`, `check_type`, `column`, `threshold`, `severity`.
> If any of these are missing, the adapter raises a `ValueError` immediately
> with a clear message listing which keys are absent.

---

## Output — Result Dict

`run_expectation()` always returns this exact shape:
```python
{
    "success":            bool,  # True = check passed, False = check failed
    "failing_count":      int,   # number of rows that failed the check
    "total_count":        int,   # total rows checked
    "unexpected_samples": list,  # up to 5 examples of failing values
}
```

### Example passing result:
```python
{
    "success":            True,
    "failing_count":      0,
    "total_count":        1000,
    "unexpected_samples": []
}
```

### Example failing result:
```python
{
    "success":            False,
    "failing_count":      42,
    "total_count":        1000,
    "unexpected_samples": [17, None, 150, 200, 15]
}
```

---

## Supported Check Types

These are all check types the adapter supports.
Passing anything outside this list raises a `ValueError`.

| check_type              | What it checks                                        | Required extra keys        |
|-------------------------|-------------------------------------------------------|----------------------------|
| `not_null`              | Column has no null (None) values                      | none                       |
| `completeness`          | Column has no null (None) values (alias of not_null)  | none                       |
| `not_empty`             | Column has no empty or whitespace-only strings        | none — regex auto-injected |
| `regex`                 | Values match a regex pattern you supply               | `regex`                    |
| `range`                 | Values fall between min and max                       | `min` and/or `max`         |
| `in_set`                | Values are in an allowed set                          | `values`                   |
| `unique`                | All values in the column are unique                   | none                       |
| `foreign_key`           | Values exist in a reference set                       | `values`                   |
| `referential_integrity` | Values exist in a reference set (alias of foreign_key)| `values`                   |
| `freshness`             | Date values fall within a time range                  | `min` and/or `max`         |

> **Notes:**
> - `not_null` and `completeness` are identical — both catch `None`/`NULL` only.
> - `not_empty` also catches `""` and `"   "` (whitespace-only strings) which `not_null` misses. You do **not** supply a `regex` key for this check — the adapter injects the correct one automatically.
> - `foreign_key` and `referential_integrity` are identical — both require that values exist in the set you provide via `values`.
> - `range` requires at least one of `min` or `max`. Passing neither raises a `RuntimeError`.
> - `threshold` is always required and is passed to GE as `mostly` internally. For example, `threshold: 0.95` means GE will pass the check if at least 95% of rows satisfy the rule.

---

## Errors the Adapter Can Raise

| Error          | When it happens                                                        |
|----------------|------------------------------------------------------------------------|
| `ValueError`   | Required key missing from check dict (name, check_type, column, etc.) |
| `ValueError`   | Unknown `check_type` not in supported list                             |
| `RuntimeError` | GE fails to create dataset, run expectation, or parse result           |

Always wrap calls in try/except in the Validation Engine:
```python
try:
    result = adapter.run_expectation(df, check)
except ValueError as e:
    # missing key or bad check_type — this is a config problem, not a data problem
    logger.error(f"Invalid check config: {e}")
except RuntimeError as e:
    # GE internal failure — log it, mark check as error, continue to next check
    logger.error(f"GE execution failed: {e}")
```

---

## Full Working Example
```python
import pandas as pd
from src.ge_adapter import GEAdapter

adapter = GEAdapter()

df = pd.DataFrame({
    "age": [25, 17, 30, None, 150, 22]
})

check = {
    "name":       "age_in_range",
    "check_type": "range",
    "column":     "age",
    "min":        18,
    "max":        120,
    "threshold":  0.95,
    "severity":   "high"
}

result = adapter.run_expectation(df, check)
print(result)
# {
#     "success":            False,
#     "failing_count":      3,       ← rows with age 17, None, and 150
#     "total_count":        6,
#     "unexpected_samples": [17, None, 150]
# }
# Note: 3 out of 6 rows failed = 50%, which is below threshold 0.95, so success=False
```

---

## Important Notes for the Validation Engine Team

1. **Never import `great_expectations` directly** — always go through
   `GEAdapter`. This is enforced by the Adapter Pattern.

2. **The adapter does not touch the database** — it is pure computation.
   The Validation Engine is responsible for writing results to the DB.

3. **All checks always run** — the adapter never stops early.
   Even if one check fails, call `run_expectation()` again for the next check.

4. **The adapter is stateless** — one `GEAdapter()` instance can be
   reused across all checks in a run. No need to create a new instance
   per check.

5. **Do not supply `regex` for `not_empty`** — the adapter injects the
   correct regex automatically. Supplying your own will override it and
   break the check.

6. **`threshold` is required, not optional** — even if you want 100% of
   rows to pass, include `"threshold": 1.0` explicitly. The adapter will
   raise a `ValueError` if it is missing.

7. **GE version warning** — built against `great-expectations==0.18.8`.
   If upgraded to GE 1.x, `ge_adapter.py` needs a full rewrite.

---

*Built by: Salma Ehab — Step 6, DataNexus DEPI 2026*
