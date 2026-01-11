# LLM Inference Benchmark

A portable benchmark tool for measuring Large Language Model inference performance on bare metal and Kubernetes.

## Features

*   **Metric Collection**: Latency (p50, p95), Throughput (tokens/sec), Warmup time.
*   **Hardware Agnostic**: Auto-detects CUDA, MPS (Apple Silicon), or CPU.
*   **Kubernetes Ready**: deployable as a Job to target specific hardware accelerators (e.g., NVIDIA L4, T4, A100).
*   **Configurable**: Easy CLI for swapping models, batch sizes, and sequence lengths.

## Prerequisites

*   Python 3.10+
*   Docker
*   `kubectl` configured for your cluster
*   `gcloud` SDK

## Local Usage (Bare Metal)

1.  **Install Dependencies**:
    ```bash
    pip install -r src/requirements.txt
    ```

2.  **Run Benchmark**:
    ```bash
    # Run on CPU/MPS/CUDA (auto-detected)
    python src/benchmark.py --model gpt2 --batch-size 1 --iterations 5
    
    # Force specific device
    python src/benchmark.py --model meta-llama/Llama-2-7b-hf --device cuda
    ```

## Kubernetes Usage

1.  **Deploy Job**:
    Use the helper script to build the image and submit the Job to your GKE cluster.
    
    ```bash
    # Default (gpt2 on nvidia-l4)
    ./k8s/deploy.sh
    
    # Custom configuration
    ./k8s/deploy.sh --model "facebook/opt-1.3b" --gpu-type "nvidia-t4" --gpu-count 1
    ```

2.  **View Results**:
    The script outputs the log command. Typically:
    ```bash
    kubectl logs -l job-name=llm-benchmark-<timestamp> -f
    ```

## Configuration

The `src/benchmark.py` script accepts the following arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--model` | `gpt2` | Hugging Face Model ID |
| `--batch-size` | `1` | Batch size for inference |
| `--input-len` | `128` | Length of input prompt (tokens) |
| `--max-new-tokens` | `128` | Number of tokens to generate |
| `--iterations` | `10` | Number of benchmark runs |
| `--device` | `auto` | `cuda`, `mps`, or `cpu` |
| `--output` | `benchmark_results.json` | Path to save JSON results |