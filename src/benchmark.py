import argparse
import time
import json
import torch
import platform
import os
from transformers import pipeline, AutoTokenizer

def get_device():
    """Determines the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def benchmark(model_id, batch_size, input_len, max_new_tokens, iterations, device):
    print(f"Loading model: {model_id} on {device}...")
    
    # Load tokenizer and model
    # trusting remote code is sometimes needed for newer models, use with caution in prod
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Ensure pad_token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        generator = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    except Exception as e:
        return {"error": str(e)}

    # Create dummy input
    # Generate a prompt of approx input_len tokens
    prompt_text = "The quick brown fox jumps over the lazy dog. " * (input_len // 9 + 1)
    # Truncate to exact token length (approx) - simplistic approach
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    if input_ids.shape[1] > input_len:
        input_ids = input_ids[:, :input_len]
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Replicate for batch size
    prompts = [prompt_text] * batch_size

    print("Warming up...")
    # Warmup
    generator(prompts, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
    print(f"Running benchmark ({iterations} iterations)...")
    latencies = []
    total_generated_tokens = 0
    
    for i in range(iterations):
        start_time = time.perf_counter()
        outputs = generator(
            prompts, 
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # usually needed for meaningful generation
            top_k=50
        )
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        latencies.append(latency)
        
        # Calculate actual generated tokens (batch_size * max_new_tokens roughly, but let's be precise if we parsed output)
        # For this benchmark, we assume max_new_tokens were generated per sequence
        total_generated_tokens += batch_size * max_new_tokens
        
        print(f"Iteration {i+1}/{iterations}: {latency:.4f}s")

    avg_latency = sum(latencies) / len(latencies)
    throughput = total_generated_tokens / sum(latencies)
    
    # Sort latencies for percentiles
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95_idx = int(len(latencies) * 0.95)
    p95 = latencies[p95_idx]

    results = {
        "config": {
            "model_id": model_id,
            "device": device,
            "batch_size": batch_size,
            "input_len": input_len,
            "max_new_tokens": max_new_tokens,
            "iterations": iterations,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
        },
        "metrics": {
            "latency_avg_sec": avg_latency,
            "latency_p50_sec": p50,
            "latency_p95_sec": p95,
            "throughput_tokens_per_sec": throughput,
            "total_tokens_generated": total_generated_tokens
        }
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model ID")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--input-len", type=int, default=128, help="Input sequence length (tokens)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of new tokens to generate")
    parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, help="Device to use (cuda, mps, cpu). Auto-detect if not set.")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    device = args.device if args.device else get_device()
    
    results = benchmark(
        args.model, 
        args.batch_size, 
        args.input_len, 
        args.max_new_tokens, 
        args.iterations, 
        device
    )
    
    print(json.dumps(results, indent=2))
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
