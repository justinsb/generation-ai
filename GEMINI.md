# GEMINI.md - Project Context & Handover

## Project Overview
**Generation AI** is a benchmarking tool designed to measure Large Language Model (LLM) inference performance. It is built to run portably on both "bare metal" (local developer workstations, VMs) and Kubernetes clusters (specifically GKE).

## Current Architecture

### 1. Core Logic (`src/benchmark.py`)
*   **Library**: Uses Hugging Face `transformers` pipelines for generation.
*   **Hardware Agnostic**: Automatically detects `cuda` (NVIDIA), `mps` (Apple Silicon), or defaults to `cpu`.
*   **Metrics**: Captures initialization time, p50/p95 latency, and token throughput.
*   **Output**: results are printed to STDOUT as JSON for easy parsing by downstream tools (logs).

### 2. Infrastructure (`k8s/`)
*   **Job-based**: The benchmark runs as a Kubernetes `Job` to ensure it runs to completion and exits.
*   **Accelerator Targeting**: Uses `nodeSelector` aiming at `cloud.google.com/gke-accelerator` to schedule pods on specific hardware (L4, T4, A100, etc.).
*   **Deployment**: A wrapper script (`deploy.sh`) handles Docker building, pushing to the current project's GCR, and templating the YAML manifest.

## Key Design Decisions
*   **Model Loading**: Currently, models are downloaded **on the fly** inside the container at runtime. This was chosen for simplicity in the prototype phase.
*   **Remote Code**: `trust_remote_code=True` is enabled in the Python script to support custom architectures (often needed for newer open-source models).

## Roadmap & Pointers for Future Agents

### 1. Model Caching (High Priority)
*   **Current State**: Every job run downloads the model weights from the Hub. This creates network overhead and startup latency.
*   **Future Task**: Implement a caching mechanism.
    *   *Kubernetes*: Use a PersistentVolumeClaim (PVC) or a HostPath (if appropriate for the cluster setup) mounted to `/root/.cache/huggingface`.
    *   *Docker*: Consider pre-baking specific popular models into the image if the set of models is small and fixed.

### 2. Metric Aggregation
*   **Current State**: JSON is logged to stdout.
*   **Future Task**: Implement a way to scrape or export these metrics (e.g., push to Prometheus, write to a GCS bucket, or a separate results aggregator service).

### 3. Optimization
*   **Quantization**: The current script loads models in `float16` (CUDA) or `float32`. Adding support for 8-bit or 4-bit quantization (via `bitsandbytes`) would be a valuable addition for running larger models on smaller GPUs.
*   **Backends**: Currently uses standard HF `pipelines`. Comparing against optimized inference servers like **vLLM** or **TGI** (Text Generation Inference) would be a natural evolution of this benchmark.

### 4. Safety
*   **Warning**: The script runs arbitrary code from Hugging Face (`trust_remote_code=True`). Ensure the `model_id` passed to the benchmark is from a trusted source.
