# Polars

Polars is a modern DataFrame library built in Rust, exposing a Python API that
is API-compatible with Pandas in concept but orders of magnitude faster for
large datasets. It is built on Apache Arrow's memory model, uses a lazy
evaluation engine that optimises query plans before execution, and exploits
SIMD and multi-core parallelism by default. For any dataset that strains Pandas,
Polars is the correct tool.

## 1. The Core Intuition (The "Why")

Pandas was designed in 2009 for datasets that comfortably fit in RAM on a
single core. Its internal architecture — a fragmented block manager, eager
evaluation that materialises intermediate results, and a single-threaded
Python GIL for most operations — scales poorly beyond a few hundred million
rows.

Polars was designed around three core principles:
1. **Apache Arrow columnar memory**: All data is stored in Arrow format, which
   is the same format used by Spark, DuckDB, and modern databases. This enables
   zero-copy interoperability across the data ecosystem.
2. **Lazy evaluation with query planning**: Operations are not immediately
   executed. Instead, they are compiled into a logical plan, optimised
   (predicate pushdown, projection pushdown, common sub-expression elimination),
   and then executed in a single efficient pass.
3. **Full parallelism by default**: Every operation that can be parallelised
   is. `groupby`, `join`, and `apply` all use all available CPU cores.

The practical result: a Polars query on 100M rows often runs in seconds where
Pandas would take minutes and require twice the memory.

## 2. The Theoretical Underpinning

### Apache Arrow Columnar Memory Format

Arrow stores each column as a contiguous buffer of fixed-width values in
native byte order. For an integer column of $n$ rows, the buffer is exactly
$n \times 4$ bytes (for `int32`). This layout enables:
- **SIMD operations**: A single CPU instruction can add, compare, or filter
  64 values simultaneously.
- **Zero-copy slicing**: A slice of a column is just a pointer + offset,
  no data copied.
- **Interoperability**: Pandas, Polars, DuckDB, and PyArrow all read the same
  buffer without copying.

### Lazy Evaluation and Query Optimisation

In lazy mode, each operation appends a node to a logical plan DAG:

$$
\text{LogicalPlan} = \text{Scan} \xrightarrow{\sigma} \text{Filter} \xrightarrow{\pi} \text{Select} \xrightarrow{\oplus} \text{Aggregate}
$$

Before execution, the optimiser applies:
- **Predicate pushdown**: move filters as early as possible to reduce rows
  early.
- **Projection pushdown**: drop columns not needed by the query before joins
  or aggregations.
- **Common sub-expression elimination**: compute shared sub-expressions once.

This can reduce memory usage by 10x and execution time by 5-50x compared to
naive eager execution.

### Expressions as First-Class Values

In Polars, an **expression** like `pl.col("revenue").mean()` is a deferred
computation, not an immediate value. Expressions can be composed:

```python
normalised = (pl.col("x") - pl.col("x").mean()) / pl.col("x").std()
```

This entire expression is compiled into a single pass over the data, with
mean and std computed once and reused. Pandas requires three separate passes
and two intermediate columns.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional


class LazyNode:
    """A node in a minimal lazy query plan.

    In a real system (Polars, Spark), these nodes are compiled to
    physical execution plans with cost-based optimisation.
    """

    def __init__(self, operation: str, parent: Optional["LazyNode"] = None,
                 **kwargs: Any) -> None:
        self.operation = operation
        self.parent    = parent
        self.kwargs    = kwargs

    def filter(self, column: str, value: Any) -> "LazyNode":
        """Append a filter (selection) node to the plan."""
        return LazyNode("filter", parent=self, column=column, value=value)

    def select(self, columns: List[str]) -> "LazyNode":
        """Append a projection node to the plan."""
        return LazyNode("select", parent=self, columns=columns)

    def describe_plan(self, indent: int = 0) -> str:
        """Recursively describe the logical plan from root to this node."""
        lines = []
        if self.parent is not None:
            lines.append(self.parent.describe_plan(indent))
        args = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        lines.append(" " * indent + f"{self.operation}({args})")
        return "\n".join(lines)


def execute_plan(
    node: LazyNode,
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate a lazy plan against a list of row dictionaries.

    In a real engine, this would be compiled to vectorised operations.
    Here we use Python lists for illustration.
    """
    # Collect operations from root to leaf
    ops = []
    current: Optional[LazyNode] = node
    while current is not None:
        ops.append(current)
        current = current.parent
    ops.reverse()

    rows = data
    for op in ops:
        if op.operation == "scan":
            pass  # already loaded
        elif op.operation == "filter":
            col, val = op.kwargs["column"], op.kwargs["value"]
            rows = [r for r in rows if r.get(col) == val]
        elif op.operation == "select":
            cols = op.kwargs["columns"]
            rows = [{c: r[c] for c in cols if c in r} for r in rows]
    return rows
```

## 4. Implementation: Production-Grade (Polars)

```python
from __future__ import annotations

from pathlib import Path
import polars as pl


def load_lazy(path: Path) -> pl.LazyFrame:
    """Load a CSV as a lazy frame; no data is read until collect()."""
    return pl.scan_csv(path)


def compute_top_groups(
    lf: pl.LazyFrame,
    group_col: str,
    value_col: str,
    top_n: int = 10,
) -> pl.DataFrame:
    """Compute top N groups by aggregate sum of a value column.

    The entire pipeline — scan, filter, groupby, sort, slice — is
    compiled into a single optimised physical plan by Polars before
    any data is read.

    Args:
        lf: Lazy frame (data not yet loaded).
        group_col: Column to group by.
        value_col: Column to sum within each group.
        top_n: Number of top groups to return.

    Returns:
        Eager DataFrame with top N groups.
    """
    return (
        lf
        .filter(pl.col(value_col) > 0)
        .group_by(group_col)
        .agg(pl.col(value_col).sum().alias("total"))
        .sort("total", descending=True)
        .limit(top_n)
        .collect()
    )


def normalise_column(
    df: pl.DataFrame,
    column: str,
) -> pl.DataFrame:
    """Z-score normalise a column using a single-pass expression.

    Polars computes mean and std in a single pass; no intermediate
    columns are created.

    Args:
        df: Source DataFrame.
        column: Column to normalise.

    Returns:
        DataFrame with an added '{column}_norm' column.
    """
    return df.with_columns(
        ((pl.col(column) - pl.col(column).mean()) / pl.col(column).std())
        .alias(f"{column}_norm")
    )


def scan_and_join(
    left_path: Path,
    right_path: Path,
    join_key: str,
) -> pl.DataFrame:
    """Lazy-scan two CSV files and join them on a key column.

    Polars pushes the join down into the scan operators when possible,
    reading only the rows needed.

    Args:
        left_path: Path to left CSV.
        right_path: Path to right CSV.
        join_key: Column name to join on.

    Returns:
        Joined DataFrame.
    """
    left  = pl.scan_csv(left_path)
    right = pl.scan_csv(right_path)
    return left.join(right, on=join_key, how="inner").collect()
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Calling `.collect()` after every operation, defeating lazy
  evaluation and forcing multiple passes over the data.

  **Fix:** Chain all operations on a `LazyFrame` and call `.collect()` exactly
  once at the end. This lets the query planner optimise the entire pipeline.

- **Mistake:** Using `.apply()` (row-wise Python UDFs) on large Polars
  DataFrames. This breaks out of Rust into Python for every row.

  **Fix:** Express computations as Polars expressions (`pl.col(...)` chains).
  For complex logic, use `map_elements` only after profiling.

- **Mistake:** Mixing Polars and Pandas DataFrames in a pipeline via
  `.to_pandas()` / `pl.from_pandas()`, which copies all data.

  **Fix:** Convert once at the boundary. If interoperability is frequent,
  use PyArrow as the intermediary to avoid copying.

- **Mistake:** Using `scan_csv` when the file is not sorted, and relying on
  streaming mode without understanding which operations are unsupported.

  **Fix:** Check `lf.explain(streaming=True)` for unsupported operations
  before deploying a streaming pipeline.

- **Mistake:** Assuming Polars has the same method names as Pandas.

  **Fix:** Consult the Polars API reference. Key differences: `.select()` vs
  `.loc`, `.with_columns()` vs `.assign()`, `.rename()` vs in-place rename.

## 6. Knowledge Check

1. **Conceptual:** Explain predicate pushdown in Polars. Why does pushing a
   filter before a join reduce both memory usage and execution time? Give a
   concrete example with row counts.

2. **Conceptual:** Polars expressions are composable and computed in a single
   pass. How does this compare to the equivalent Pandas code that computes
   mean, std, and the normalised column in three separate operations?

3. **Coding challenge:** Using Polars lazy mode, read a 10M-row CSV with
   columns `["user_id", "product_id", "price", "quantity"]`, compute the total
   revenue per product, and return the top 5 products — all without loading the
   full dataset into memory at once.

## References

1. Polars User Guide: https://docs.pola.rs
2. Apache Arrow Columnar Format: https://arrow.apache.org/docs/format/Columnar.html
3. Vogelgesang, R. (2023). Polars — Lightning-fast DataFrame library for Rust and Python. GitHub: https://github.com/pola-rs/polars
4. McKinney, W. (2017). *Python for Data Analysis* (2nd ed.). O'Reilly. *(Pandas background.)*
