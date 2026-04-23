# Pandas

Pandas is the standard Python library for tabular data manipulation. Its
DataFrame abstraction provides a labelled, two-dimensional table with
heterogeneous column types, and its API expresses relational algebra
operations in a concise chainable syntax. For ML engineers, proficiency with
Pandas is proficiency with the preprocessing layer every structured-data
model depends on.

## 1. The Core Intuition (The "Why")

Before Pandas, tabular data manipulation in Python meant SQL queries against
a database, CSV shell scripts, or nested loops over lists of dictionaries.
All were slow, error-prone, or required leaving Python entirely.

Wes McKinney (2010) created Pandas to bring the expressiveness of R's
`data.frame` to Python, representing a table as a collection of aligned
Series (typed NumPy arrays). This provided fast columnar operations via NumPy
vectorisation, labelled axes, automatic alignment of differently-indexed data,
and first-class handling of missing values.

Understanding Pandas deeply — especially copy-versus-view semantics and memory
layout — separates engineers who handle 10M-row datasets from those who run
out of memory or spend hours debugging silent data mutations.

## 2. The Theoretical Underpinning

### DataFrames as Relational Algebra

A Pandas DataFrame implements a **relation** from the relational model. The
five fundamental relational operations all have Pandas equivalents:

| Relational operation | Pandas equivalent |
|----------------------|-------------------|
| Selection $\sigma$   | `df[df.col > x]` |
| Projection $\pi$     | `df[["a", "b"]]` |
| Cartesian product    | `df.merge(other, how="cross")` |
| Natural join         | `df.merge(other, on="key")` |
| Aggregation          | `df.groupby("key").agg(...)` |

### Memory Layout: Column-Oriented Storage

A DataFrame stores each column as a contiguous typed NumPy array. For a query
that filters on column $C$ and sums column $V$, only those two arrays need to
be read from memory. This column-oriented layout makes analytical queries fast
but makes row-wise Python `apply` calls slow (100-1000x overhead vs vectorised
operations).

### Views and Copies

```python
subset = df[df["col"] > 5]   # might be a view OR a copy
subset["new_col"] = 1        # SettingWithCopyWarning
```

Whether the result is a view or a copy depends on the indexing operation and
internal block structure. A view propagates modifications back to the original
DataFrame silently. Always use `.loc` or `.assign()` for mutation.

### GroupBy as Split-Apply-Combine

The `groupby` pattern:
1. **Split**: partition rows into groups by key(s).
2. **Apply**: compute a reduction on each group.
3. **Combine**: assemble the results.

$$\text{result} = \bigoplus_{k \in K} f\!\left(\{r \in R : r.\text{key} = k\}\right)$$

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations
from typing import Any, Dict, List


class MiniDataFrame:
    """Minimal columnar DataFrame demonstrating select, filter, groupby-sum."""

    def __init__(self, data: Dict[str, List[Any]]) -> None:
        lengths = {len(v) for v in data.values()}
        if len(lengths) > 1:
            raise ValueError("All columns must have the same length.")
        self._data = {k: list(v) for k, v in data.items()}

    def select(self, columns: List[str]) -> "MiniDataFrame":
        """Relational projection onto a column subset."""
        missing = [c for c in columns if c not in self._data]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        return MiniDataFrame({c: self._data[c] for c in columns})

    def filter_equal(self, column: str, value: Any) -> "MiniDataFrame":
        """Keep rows where column == value."""
        indices = [i for i, v in enumerate(self._data[column]) if v == value]
        return MiniDataFrame({
            col: [vals[i] for i in indices]
            for col, vals in self._data.items()
        })

    def groupby_sum(self, key_col: str, value_col: str) -> "MiniDataFrame":
        """Sum value_col within each key_col group."""
        sums: Dict[Any, float] = {}
        for k, v in zip(self._data[key_col], self._data[value_col]):
            sums[k] = sums.get(k, 0.0) + v
        return MiniDataFrame({key_col: list(sums.keys()),
                               value_col: list(sums.values())})
```

## 4. Implementation: Production-Grade (Pandas)

```python
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_and_validate(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV and validate that required columns are present."""
    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def safe_assign(df: pd.DataFrame, column: str, values: pd.Series) -> pd.DataFrame:
    """Return a new DataFrame with column set to values.

    Uses .assign() to avoid SettingWithCopyWarning. Always produces
    a copy, so the original DataFrame is never mutated.
    """
    return df.assign(**{column: values})


def aggregate_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    agg_map: dict[str, str],
) -> pd.DataFrame:
    """Group rows and apply per-column aggregation functions.

    Example:
        aggregate_by_group(df, ["category"], {"revenue": "sum"})
    """
    return df.groupby(group_cols).agg(agg_map).reset_index()


def memory_usage_mb(df: pd.DataFrame) -> pd.Series:
    """Return per-column memory usage in megabytes."""
    return df.memory_usage(deep=True) / (1024 ** 2)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using `.apply(func, axis=1)` on DataFrames with millions of rows.

  **Fix:** Use vectorised NumPy or Pandas column operations. For truly
  non-vectorisable logic, use Polars `map_elements` or Numba.

- **Mistake:** Chaining `[][]` for assignment triggers `SettingWithCopyWarning`
  and may silently do nothing on the original.

  **Fix:** Use `.loc[mask, "col"] = value` for in-place assignment or `.assign()`
  for functional transformation.

- **Mistake:** Loading a 5 GB CSV with default dtypes, consuming 15+ GB RAM
  because string columns are stored as Python objects.

  **Fix:** Specify `dtype=` in `read_csv`. Use `category` for low-cardinality
  strings; `int32`/`float32` where full precision is not needed.

- **Mistake:** Iterative merges inside a loop when a single multi-key merge works.

  **Fix:** Chain `.merge()` calls. Sort on merge keys for large tables first.

- **Mistake:** Using Pandas for data exceeding available RAM.

  **Fix:** Use Polars (lazy mode), DuckDB, or PySpark for out-of-core or
  distributed processing.

## 6. Knowledge Check

1. **Conceptual:** Explain the difference between a Pandas view and a copy.
   Why does `SettingWithCopyWarning` exist, and why can the same expression
   return a view sometimes and a copy other times?

2. **Conceptual:** Describe how you would implement a custom per-group
   transformation that cannot be expressed as a built-in aggregation, and
   what the performance implications are compared to `.groupby().agg()`.

3. **Coding challenge:** Given a DataFrame with columns
   `["user_id", "event_type", "timestamp", "revenue"]`, compute the
   revenue-per-event-type for the top 10 users by total revenue using only
   Pandas operations (no Python loops).

## References

1. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of SciPy 2010*.
2. McKinney, W. (2011). pandas: A Foundational Python Library for Data Analysis and Statistics. *PyHPC 2011*.
3. Codd, E. F. (1970). A relational model of data for large shared data banks. *CACM*, 13(6), 377-387.
