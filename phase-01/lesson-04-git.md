# Git

Git is a distributed, content-addressed version control system that makes
software and ML projects reproducible, reviewable, and recoverable. In a
field where the difference between experiment A and experiment B might be a
one-line change to a preprocessing script, the ability to precisely trace,
compare, and revert state is not optional — it is the foundation of scientific
rigour.

## 1. The Core Intuition (The "Why")

Before version control, the standard practice was copying entire project
directories: `model_v1/`, `model_v2_final/`, `model_v2_ACTUALLY_FINAL/`.
This approach fails immediately with more than one person. It also fails for a
single engineer who needs to understand, three months later, which
configuration produced the best result.

Linus Torvalds created Git in 2005 to manage the Linux kernel — thousands of
contributors, where any change could interact with any other. The required
properties were: fast local operations, cryptographic integrity, cheap
branching, and no single point of failure. Every one of these properties
matters directly for ML engineering:

- **Fast local operations:** Branching, diffing, and reverting do not require
  network access, so iterating on experiments is instant.
- **Cryptographic integrity:** A commit hash is a verifiable fingerprint of
  the entire project state at that point in time.
- **Cheap branching:** Testing a new loss function does not touch the
  production training code.
- **No single point of failure:** Every clone is a full backup.

## 2. The Theoretical Underpinning

### Content-Addressed Object Store

The core insight of Git is that objects are identified by their *content*, not
their location. Every blob (file), tree (directory), and commit is identified
by the SHA-1 hash of its byte payload:

$$
\text{id}(x) = \operatorname{SHA1}(\text{bytes}(x))
$$

SHA-1 maps any byte sequence to a 160-bit digest. Two objects are identical
if and only if their hashes collide — practically impossible accidentally
(Git is migrating toward SHA-256 for collision-resistance).

The consequence: the object store is **immutable by design**. You cannot change
a committed file without changing its hash, which changes the tree hash, which
changes the commit hash. A commit hash is a stable, verifiable reference to the
*entire* state of the project at a point in time.

### The Commit DAG

A commit packages a tree snapshot with metadata:

$$
c_t = \operatorname{SHA1}\!\left(\text{tree}(T_t),\; c_{t-1},\; \text{message},\; \text{author},\; \text{timestamp}\right)
$$

Parent pointers create a **directed acyclic graph** of commits:

$$
G = (V, E), \quad V = \{c_t\},\quad E = \{(c_t, c_{t-1})\}
$$

Branches are just named pointers to leaves of this graph. Merging finds a
common ancestor and combines two descendant states.

### Merge Conflicts

Let $\Delta_a(p)$ and $\Delta_b(p)$ denote changes to path $p$ on branches
$a$ and $b$ relative to their common ancestor. A conflict exists when

$$
\Delta_a(p) \neq \varnothing \quad\text{and}\quad
\Delta_b(p) \neq \varnothing \quad\text{and}\quad
\Delta_a(p) \neq \Delta_b(p)
$$

Git cannot resolve this automatically because the intent behind both changes is
unknown. The human must decide which edit is correct.

### Why This Matters for ML

"Code" in ML is not just source files. It includes preprocessing scripts
(small changes, huge downstream effects), hyperparameter configs (a learning
rate change makes results incomparable), and evaluation code (a bug here makes
a bad model look good). Git's content-addressing provides ground truth for
which version of each was used in any given experiment.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Mapping, Optional


def hash_object(payload: bytes) -> str:
    """SHA-1 of raw bytes — mirrors Git object identity."""
    return hashlib.sha1(payload).hexdigest()


def build_tree_snapshot(files: Mapping[str, str]) -> Dict[str, str]:
    """Map file paths to SHA-1 hashes of their contents."""
    return {
        path: hash_object(content.encode("utf-8"))
        for path, content in sorted(files.items())
    }


@dataclass
class MiniCommit:
    tree: Dict[str, str]
    parent: Optional[str]
    message: str
    author: str
    commit_hash: str


def create_commit(
    tree: Mapping[str, str],
    parent: Optional[str],
    message: str,
    author: str,
) -> MiniCommit:
    """Hash a new commit object.

    The commit hash is deterministic: identical tree + parent + message
    + author always produce the same hash — making every commit verifiable.
    """
    payload = json.dumps(
        {"tree": dict(tree), "parent": parent,
         "message": message, "author": author},
        sort_keys=True,
    ).encode("utf-8")
    return MiniCommit(
        tree=dict(tree),
        parent=parent,
        message=message,
        author=author,
        commit_hash=hash_object(payload),
    )


def detect_merge_conflicts(
    base: Mapping[str, str],
    left: Mapping[str, str],
    right: Mapping[str, str],
) -> Dict[str, str]:
    """Identify paths changed differently on two branches.

    Returns a dict of conflicting paths with explanations.
    """
    conflicts: Dict[str, str] = {}
    for path in sorted(set(base) | set(left) | set(right)):
        left_changed  = left.get(path)  != base.get(path)
        right_changed = right.get(path) != base.get(path)
        both_differ   = left.get(path)  != right.get(path)
        if left_changed and right_changed and both_differ:
            conflicts[path] = "Both branches modified this path differently."
    return conflicts
```

## 4. Implementation: Production-Grade (Git CLI)

```bash
# Initialise and first commit
git init
git switch -c main
git add .
git commit -m "feat: initialise project scaffold"

# Branch-based experiment workflow
git switch -c experiment/dropout-0.2
# ... edit files ...
git add src/model.py configs/train.yaml
git commit -m "experiment: add dropout=0.2 to transformer block"

# Merge back with a commit that preserves history
git switch main
git merge --no-ff experiment/dropout-0.2 \
    -m "merge: incorporate dropout experiment results"

# Tag the reproducible checkpoint used for a paper evaluation
git tag -a v1.0-eval -m "checkpoint: paper evaluation results"

# Push everything including tags
git push origin main --follow-tags
```

**For large binary artefacts** (model weights, dataset shards): use Git LFS,
DVC, or MLflow. The rule: anything binary that changes infrequently can go in
LFS; anything generated should not be tracked at all.

```bash
# Track model checkpoints with Git LFS
git lfs track "*.pt" "*.ckpt" "*.safetensors"
git add .gitattributes
git commit -m "chore: configure Git LFS for model artefacts"
```

## 5. Production Pitfalls & Pro-Tips

- ❌ **The mistake:** Committing model weights, dataset shards, or experiment
  outputs to the repository, inflating it to gigabytes.
  ✅ **The fix:** Add `*.pt`, `*.ckpt`, `*.parquet`, and `data/` to
  `.gitignore`. Use Git LFS for artefacts that genuinely need version tracking.

- ❌ **The mistake:** Committing config files that contain API keys, database
  passwords, or model-serving tokens.
  ✅ **The fix:** Store secrets in environment variables or a secrets manager.
  Use `git-secrets` or `trufflehog` to scan history; add pre-commit hooks to
  block future leaks.

- ❌ **The mistake:** Using long-lived feature branches that diverge from main
  for weeks, producing enormous merge conflicts that are impossible to review.
  ✅ **The fix:** Trunk-based development — merge small, focused changes into
  main daily. Use feature flags to hide incomplete work in production.

- ❌ **The mistake:** Writing commit messages like "fix stuff" or "wip" that
  communicate nothing about intent, making it impossible to debug regressions.
  ✅ **The fix:** Follow Conventional Commits: `type(scope): description`.
  Examples: `feat(trainer): add gradient accumulation`,
  `fix(dataloader): handle empty last batch`.

- ❌ **The mistake:** Using `git push --force` on shared branches without
  understanding its implications.
  ✅ **The fix:** `--force` rewrites published history and will break every
  teammate's working copy. Use `--force-with-lease` to check that you are not
  clobbering someone else's recent push.

## 6. Knowledge Check

1. **Conceptual:** Git identifies every object by its SHA-1 hash. Explain why
   this "content-addressed" design makes commits tamper-evident, and describe
   one ML engineering scenario where this property provides direct value.

2. **Conceptual:** Two branches may merge without conflicts yet still produce
   incorrect joint behaviour — one branch changed the model architecture and
   another changed the training loop to assume the old architecture. How would
   you detect and prevent these "semantic conflicts"?

3. **Coding challenge:** In 20 lines of Python, write a function that takes a
   `{path: content}` dictionary and returns a SHA-1 tree hash that changes if
   and only if any file content changes — mirroring Git's tree object hashing.

## References

1. Torvalds, L. (2005). Git initial announcement, Linux Kernel Mailing List.
2. Chacon, S., & Straub, B. (2014). *Pro Git* (2nd ed.). Apress. https://git-scm.com/book
3. Conventional Commits specification: https://www.conventionalcommits.org
