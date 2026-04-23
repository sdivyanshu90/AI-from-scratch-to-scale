# Orchestration: Docker, Kubernetes, and Ray

Deploying a trained model into production requires more than running a single
Python process. Container orchestration handles packaging, scaling, fault
tolerance, and resource management across distributed fleets of machines.
Docker standardises the deployment unit; Kubernetes (Burns et al., 2016)
orchestrates containers at cluster scale; Ray (Moritz et al., 2018) provides
a distributed computing framework purpose-built for ML workloads.

## 1. The Core Intuition (The "Why")

A model that runs perfectly on your laptop may fail on a production server
due to dependency differences, memory limits, or CUDA version mismatches.
Docker solves this by bundling the application and all its dependencies into
an immutable container image. The image runs identically on any host that
has the Docker runtime.

Once you have containers, you need to manage hundreds of them: health checks,
rolling updates, auto-scaling, and failover. Kubernetes automates this.
It treats compute resources as a pool, schedules containers (Pods) onto nodes
based on requested CPU/memory/GPU, and restarts failed containers automatically.

Ray Serve adds a higher-level ML abstraction: you define model deployments
as Python classes, set `num_replicas`, and Ray manages the routing, batching,
and scaling. Ray is particularly useful for ML pipelines where different
stages (preprocessing, inference, post-processing) have different
computational requirements and need to scale independently.

## 2. The Theoretical Underpinning

### Container Layering (Union Filesystem)

A Docker image is a stack of read-only **layers** combined by a union
filesystem (overlay2). Each Dockerfile instruction adds a layer. The
final container adds a thin writable layer on top. Images are efficiently
stored and transferred because identical layers are shared between images.

Layer $L_i$ stores only the filesystem delta from $L_{i-1}$:

$$\text{Image} = L_1 \cup L_2 \cup \cdots \cup L_n$$

### Capacity Planning

For an LLM serving cluster with request arrival rate $\lambda$ requests/s
and mean service time $1/\mu$ seconds/request per replica, the minimum
number of replicas to avoid queue build-up (using Little's Law) is:

$$n \geq \left\lceil \frac{\lambda}{\mu} \right\rceil$$

A Kubernetes Horizontal Pod Autoscaler (HPA) adjusts $n$ dynamically based
on CPU/memory/custom metrics (e.g., request queue depth).

### Resource Placement Constraint

The Kubernetes scheduler ensures each Node $k$ is not over-committed:

$$\sum_{i \in \text{Pods}(k)} r_i \leq C_k$$

where $r_i$ is the resource request (CPU, memory, GPU) of Pod $i$ and
$C_k$ is the capacity of Node $k$. For GPU workloads, $r_i$ typically
requests exactly 1 GPU (LLMs do not time-share GPUs easily).

### Rolling Updates

A Kubernetes rolling update replaces old replicas with new ones gradually:

- `maxSurge`: Extra replicas above desired count during update
- `maxUnavailable`: Replicas below desired count during update

For zero-downtime deployment, set `maxUnavailable: 0` and `maxSurge: 1`.
This guarantees the desired number of replicas are serving at all times.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Resource:
    """Computational resources required or available.

    Attributes:
        cpu_cores:  Number of CPU cores.
        memory_gb:  Memory in gigabytes.
        gpu_count:  Number of GPU devices.
    """
    cpu_cores: float
    memory_gb: float
    gpu_count: int = 0


@dataclass
class Task:
    """A unit of work to schedule onto a worker.

    Attributes:
        task_id:   Unique identifier.
        required:  Computational resources this task needs.
    """
    task_id:  str
    required: Resource


@dataclass
class Worker:
    """A compute node with a fixed resource capacity.

    Attributes:
        worker_id: Unique identifier.
        capacity:  Total resources on this node.
        used:      Resources currently allocated.
    """
    worker_id: str
    capacity:  Resource
    used:      Resource = field(default_factory=lambda: Resource(0.0, 0.0, 0))

    def can_place(self, task: Task) -> bool:
        """Return True if this worker has enough free resources.

        Checks all three resource dimensions independently.

        Args:
            task: The task to check placement for.

        Returns:
            Whether the task can be placed on this worker.
        """
        return (
            self.used.cpu_cores + task.required.cpu_cores <= self.capacity.cpu_cores
            and self.used.memory_gb + task.required.memory_gb <= self.capacity.memory_gb
            and self.used.gpu_count + task.required.gpu_count <= self.capacity.gpu_count
        )

    def allocate(self, task: Task) -> None:
        """Reserve resources for a task."""
        self.used.cpu_cores += task.required.cpu_cores
        self.used.memory_gb += task.required.memory_gb
        self.used.gpu_count += task.required.gpu_count


def schedule_tasks(tasks: list[Task], workers: list[Worker]) -> dict[str, str]:
    """First-fit bin packing: place each task on the first worker that fits.

    This is a simplified version of the bin-packing problem that
    Kubernetes solves during Pod scheduling. Kubernetes uses a more
    sophisticated scoring function, but first-fit captures the essence.

    Args:
        tasks:   Tasks to schedule.
        workers: Available workers with their capacities.

    Returns:
        Dict mapping task_id to worker_id for each successfully placed task.
    """
    placement: dict[str, str] = {}
    for task in tasks:
        for worker in workers:
            if worker.can_place(task):
                worker.allocate(task)
                placement[task.task_id] = worker.worker_id
                break
    return placement
```

## 4. Implementation: Production-Grade (Ray Serve + FastAPI)

```python
from __future__ import annotations

import asyncio
from typing import Any

import ray
from ray import serve
from fastapi import FastAPI


app = FastAPI()


@serve.deployment(
    num_replicas     = 2,          # Start with 2 replicas; HPA scales dynamically
    ray_actor_options = {
        "num_gpus": 1,             # Each replica owns 1 GPU
        "num_cpus": 4,
    },
    max_ongoing_requests = 10,     # Queue backpressure per replica
)
@serve.ingress(app)
class LLMService:
    """Ray Serve deployment wrapping a language model.

    Ray Serve handles routing, load balancing, and replica management.
    The @serve.ingress decorator mounts FastAPI onto the deployment,
    so HTTP requests are automatically routed to the correct replica.
    """

    def __init__(self, model_name: str = "microsoft/phi-2") -> None:
        from transformers import pipeline
        # Each replica loads the model independently onto its assigned GPU
        self.pipe = pipeline("text-generation", model=model_name, device=0)

    @app.post("/generate")
    async def generate(self, request: dict[str, Any]) -> dict[str, str]:
        """Generate text for a single prompt.

        Args:
            request: JSON body with keys: prompt (str), max_tokens (int).

        Returns:
            JSON with key: generated_text (str).
        """
        prompt     = request["prompt"]
        max_tokens = request.get("max_tokens", 256)
        # Run CPU-bound pipeline in a thread pool to avoid blocking the event loop
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.pipe(prompt, max_new_tokens=max_tokens)
        )
        return {"generated_text": result[0]["generated_text"]}


def build_ray_application() -> serve.Application:
    """Build the Ray Serve application from an environment variable.

    In production, override MODEL_NAME via the Serve config YAML or
    an environment variable injected by Kubernetes Secrets.

    Returns:
        Bound Ray Serve application.
    """
    import os
    model_name = os.environ.get("MODEL_NAME", "microsoft/phi-2")
    return LLMService.bind(model_name=model_name)


# Kubernetes Deployment YAML equivalent (shown here as documentation):
# ---
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   name: llm-service
# spec:
#   replicas: 2
#   selector:
#     matchLabels: {app: llm-service}
#   template:
#     metadata:
#       labels: {app: llm-service}
#     spec:
#       containers:
#       - name: llm-service
#         image: myregistry/llm-service:v1.2.0
#         resources:
#           requests: {memory: "16Gi", cpu: "4", nvidia.com/gpu: "1"}
#           limits:   {memory: "24Gi", cpu: "8", nvidia.com/gpu: "1"}
#         env:
#         - name: MODEL_NAME
#           valueFrom:
#             secretKeyRef: {name: model-config, key: model_name}
#         livenessProbe:
#           httpGet: {path: /healthz, port: 8000}
#           initialDelaySeconds: 60
#         readinessProbe:
#           httpGet: {path: /readyz, port: 8000}
#           initialDelaySeconds: 60
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Building Docker images with `FROM python:3.11` and installing
  CUDA inside the image. This produces an enormous, slow-to-build image where
  the CUDA version is baked in and cannot be changed without a full rebuild.

  **Fix:** Use `FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04` as the base and
  install only the Python packages on top. Separate the CUDA base layer
  (changes rarely) from the application layer (changes often) to maximise
  Docker layer cache hit rate.

- **Mistake:** Setting Kubernetes CPU/memory requests equal to limits
  (guaranteed QoS) for LLM serving containers. GPU memory is the binding
  constraint, but CPU and RAM usage fluctuates; tight limits cause OOM kills
  during prompt-heavy traffic spikes.

  **Fix:** Set memory requests slightly below limits and CPU requests
  conservatively. Set GPU requests = limits = 1 (GPU sharing is
  not supported for most LLM workloads). Monitor actual usage with
  `kubectl top pods` before finalising resource quotas.

- **Mistake:** Deploying a new model version with `kubectl set image` and
  not verifying readiness probes. If the readiness probe passes before the
  model is fully loaded (model loading can take 60-120s for large models),
  the HPA routes live traffic to an unready replica.

  **Fix:** Implement a `/readyz` endpoint that returns 200 only after the
  model weights are fully loaded and a warmup inference has completed
  successfully. Set `initialDelaySeconds` to at least the expected load time.

- **Mistake:** Using a single Ray head node without fault tolerance. If the
  head node fails, the entire cluster state is lost and all in-flight requests
  fail.

  **Fix:** Enable Ray GCS fault tolerance by running the Global Control
  Store (GCS) on a separate Redis instance. Use `ray up` with a fault-tolerant
  cluster config or use KubeRay, which manages Ray clusters as Kubernetes
  custom resources.

- **Mistake:** Scaling replicas based on CPU utilisation only. LLM workloads
  are GPU-bound; CPU utilisation stays low even when the GPU is saturated
  and requests are queueing.

  **Fix:** Expose a custom metric (e.g., request queue depth, P95 latency)
  to the Kubernetes metrics server and configure the HPA `customMetrics`
  field to scale on that metric instead of CPU.

## 6. Knowledge Check

1. **Conceptual:** Explain the Kubernetes control loop. What is the difference
   between a Deployment and a Pod? Why does Kubernetes restart Pods that
   fail health checks, and how does this differ from process supervision
   (e.g., systemd)?

2. **Conceptual:** Derive the minimum number of replicas needed to serve
   a request rate of 100 req/s if each replica processes 15 req/s on average.
   How does the Horizontal Pod Autoscaler use this relationship, and what
   metric should drive scaling for LLM workloads?

3. **Coding challenge:** Implement a bin-packing scheduler from scratch.
   Given 20 tasks with random resource requirements (0-4 CPUs, 0-8 GB RAM,
   0-1 GPU) and 5 workers each with (16 CPUs, 64 GB RAM, 2 GPUs), run
   first-fit and best-fit scheduling. Report how many tasks are unscheduled
   under each policy and the average resource utilisation per worker.

## References

1. Merkel, D. (2014). Docker: Lightweight Linux containers for consistent development and deployment. *Linux Journal*, 2014(239).
2. Burns, B., Grant, B., Oppenheimer, D., Brewer, E., & Wilkes, J. (2016). Borg, Omega, and Kubernetes. *ACM Queue*, 14(1).
3. Moritz, P., Nishihara, R., Wang, S., et al. (2018). Ray: A distributed framework for emerging AI applications. *OSDI 2018*. arXiv:1712.05889.
4. Anyscale. (2023). Ray Serve documentation. https://docs.ray.io/en/latest/serve/index.html.
