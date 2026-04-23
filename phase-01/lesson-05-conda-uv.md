# Conda and uv: Environment and Package Management

Reproducible environments are as important as reproducible code. A model that
trains perfectly in one Python environment but silently breaks in another is
not a reliable system — it is an experiment that happened to work once.
Conda and uv solve different but complementary parts of the environment
management problem, and understanding what each does well determines when to
use which.

## 1. The Core Intuition (The "Why")

Python's default package management (`pip` with a global site-packages
directory) works for simple projects. It fails for ML engineering for two
reasons:

1. **Dependency conflicts.** A data-science project typically depends on
   dozens of packages that each have their own version requirements. Installing
   a new package can silently break a previously working environment by
   upgrading a shared dependency that another package needed pinned at an
   older version.

2. **Non-Python dependencies.** CUDA, cuDNN, BLAS, and other native libraries
   are not Python packages. `pip` cannot install or manage them. Conda was
   built specifically to handle the mixed Python/native dependency graph that
   ML workloads require.

The solution is **environment isolation**: each project gets its own
independent set of packages and, in Conda's case, native libraries. Any
environment can be fully reproduced on any machine from a lock file.

The more recent `uv` tool (by Astral) is a Rust-based replacement for the
pip/virtualenv/pip-tools stack that achieves 10-100× speed improvements by
using a SAT solver for dependency resolution and parallelised downloads.
It is ideal for pure-Python tooling; Conda remains the standard for managing
CUDA/cuDNN alongside Python.

## 2. The Theoretical Underpinning

### Dependency Resolution as a SAT Problem

Installing a set of packages $P$ such that all version constraints are
simultaneously satisfied is a version of the **Boolean satisfiability problem
(SAT)**. Each package version is a Boolean variable; the version constraints
between packages form the clauses.

Formally, let $p_{i,v}$ be the variable "package $i$ is installed at version
$v$". The feasibility constraint for a dependency "package $a$ at version $v$
requires package $b$ at version in range $R$" is:

$$
p_{a,v} \Rightarrow \bigvee_{v' \in R} p_{b,v'}
$$

A solution to the full set of such implications is a valid environment. Pip's
old greedy resolver found this solution heuristically and could produce
inconsistent environments without warnings. Both Conda's `mamba` resolver and
uv's resolver use proper SAT-based solving, which either finds a consistent
solution or produces a clear conflict report.

### Virtual Environments

A virtual environment is a self-contained directory tree containing a Python
interpreter and its own `site-packages` directory. The shell `PATH` is
modified so that `python` and `pip` resolve to the environment-local versions.
Because each project has its own environment tree, there are no global
dependency conflicts.

### Lock Files

A **lock file** records the exact version and hash of every installed package,
including transitive dependencies. This is the difference between
*reproducible by specification* (a `requirements.txt` with approximate
versions) and *reproducible by proof* (a lock file where every package hash
is verified on install). `uv lock` and `conda-lock` generate lock files.
Commit lock files to version control.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional


@dataclass
class PackageRequirement:
    """A versioned package dependency constraint."""
    name: str
    min_version: Optional[str] = None   # e.g. "1.2.0"
    max_version: Optional[str] = None   # exclusive upper bound


@dataclass
class ResolvedEnvironment:
    """A fully-resolved set of package versions with no conflicts."""
    packages: Dict[str, str] = field(default_factory=dict)  # name -> version


def parse_requirements(lines: List[str]) -> List[PackageRequirement]:
    """Parse a simple requirements.txt into structured constraints.

    Handles lines like: 'numpy>=1.24,<2.0' and 'torch==2.1.0'.

    Args:
        lines: List of requirement strings.

    Returns:
        List of PackageRequirement objects.
    """
    reqs: List[PackageRequirement] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Very simplified parsing for illustration
        if ">=" in line:
            name, version = line.split(">=", 1)
            reqs.append(PackageRequirement(name.strip(), min_version=version.strip()))
        elif "==" in line:
            name, version = line.split("==", 1)
            reqs.append(PackageRequirement(name.strip(), min_version=version.strip(),
                                           max_version=version.strip()))
        else:
            reqs.append(PackageRequirement(line))
    return reqs


def check_conflicts(env: ResolvedEnvironment,
                    reqs: List[PackageRequirement]) -> List[str]:
    """Report which requirements are violated by a resolved environment.

    In a real resolver this is computed as part of SAT solving.
    Here we do a simple post-hoc check for illustration.

    Args:
        env: Resolved environment with exact package versions.
        reqs: List of requirements to validate.

    Returns:
        List of human-readable conflict descriptions.
    """
    conflicts: List[str] = []
    for req in reqs:
        installed = env.packages.get(req.name)
        if installed is None:
            conflicts.append(f"{req.name}: not installed")
        elif req.min_version and installed < req.min_version:
            conflicts.append(
                f"{req.name}: installed {installed} < required {req.min_version}"
            )
        elif req.max_version and installed > req.max_version:
            conflicts.append(
                f"{req.name}: installed {installed} > maximum {req.max_version}"
            )
    return conflicts
```

## 4. Implementation: Production-Grade (Conda + uv CLI)

```bash
# ─── Conda workflow ──────────────────────────────────────────────────────────
# Create a new environment for a PyTorch project with CUDA 12.1
conda create -n ml-project python=3.11 pytorch torchvision \
    pytorch-cuda=12.1 -c pytorch -c nvidia

# Activate and verify GPU access
conda activate ml-project
python -c "import torch; print(torch.cuda.is_available())"

# Export a reproducible lock file (includes native libraries)
conda env export > environment.lock.yaml

# Recreate the exact environment anywhere
conda env create -f environment.lock.yaml


# ─── uv workflow (pure Python, much faster) ──────────────────────────────────
# Initialise a project with pyproject.toml
uv init my-project && cd my-project

# Add dependencies (resolves in milliseconds, not minutes)
uv add torch transformers datasets accelerate

# Lock exact versions to lock file (commit this)
uv lock

# Install from lock file on CI / another machine
uv sync --frozen

# Run a script in the project environment without explicit activation
uv run python train.py
```

## 5. Production Pitfalls & Pro-Tips

- ❌ **The mistake:** Installing packages into the base Conda environment,
  which gradually becomes a graveyard of conflicting versions that are
  impossible to debug.
  ✅ **The fix:** Always create a named environment per project. Treat the
  base environment as read-only.

- ❌ **The mistake:** Committing only `requirements.txt` with approximate
  version ranges (`>=1.2`). Two installs one month apart can produce different
  environments silently.
  ✅ **The fix:** Commit a lock file (`uv.lock`, `conda-lock.yml`, or
  `requirements.txt` generated by `pip freeze`). Verify reproducibility on CI.

- ❌ **The mistake:** Using `pip install` inside a Conda environment for
  packages that have Conda-native builds (PyTorch, numpy, scipy). Pip builds
  often link against different BLAS/LAPACK versions than Conda builds, causing
  subtle performance and correctness differences.
  ✅ **The fix:** Install everything possible via Conda first, then use pip
  for any packages not available on conda-forge or the pytorch channel.

- ❌ **The mistake:** Hard-coding the full Conda environment path (`/opt/conda/
  envs/myenv/bin/python`) in scripts. When the environment is moved or
  recreated, the path breaks.
  ✅ **The fix:** Always activate the environment and use `python` (not the
  absolute path). In containerised environments, add the environment's `bin/`
  to `PATH` in the Dockerfile.

- ❌ **The mistake:** Forgetting that Python version mismatches (3.10 vs 3.11)
  can cause subtle behavioural differences (e.g., dict ordering, exception
  groups) that affect training reproducibility.
  ✅ **The fix:** Pin the Python version in the lock file and in CI.

## 6. Knowledge Check

1. **Conceptual:** Explain why package dependency resolution is an NP-hard
   problem in the general case and why early resolvers produced inconsistent
   environments silently while modern ones (mamba, uv) report conflicts
   explicitly.

2. **Conceptual:** What is the difference between a `requirements.txt` with
   pinned versions and a lock file with cryptographic hashes? In what scenario
   does the difference matter most?

3. **Coding challenge:** Write a Python function that takes a list of `name==version`
   strings and checks whether any package appears twice with different versions.
   Return the set of conflicting package names.

## References

1. Conda documentation: https://docs.conda.io
2. uv documentation: https://docs.astral.sh/uv
3. PEP 517 – A build-system independent format for source trees: https://peps.python.org/pep-0517
4. PEP 660 – Editable installs for pyproject.toml based builds: https://peps.python.org/pep-0660
5. Jangda, A., et al. (2019). Not all versions are equal: Analysing the dependency networks of software ecosystems. *MSR 2019*.
