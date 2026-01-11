#!/bin/bash
set -e

# Default values
MODEL_ID="gpt2"
GPU_TYPE="nvidia-l4"
GPU_COUNT="1"
JOB_SUFFIX=$(date +%s)

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_ID="$2"; shift ;;
        --gpu-type) GPU_TYPE="$2"; shift ;;
        --gpu-count) GPU_COUNT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ID=$(gcloud config get project)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No Google Cloud project configured."
    exit 1
fi

IMAGE_URI="gcr.io/${PROJECT_ID}/llm-benchmark:latest"

echo "Building and pushing image to ${IMAGE_URI}..."
docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}

echo "Deploying benchmark job..."
sed -e "s|"${IMAGE_URI}"|${IMAGE_URI}|g" \
    -e "s|"${MODEL_ID}"|${MODEL_ID}|g" \
    -e "s|"${GPU_TYPE}"|${GPU_TYPE}|g" \
    -e "s|"${GPU_COUNT}"|${GPU_COUNT}|g" \
    -e "s|"${JOB_SUFFIX}"|${JOB_SUFFIX}|g" \
    k8s/benchmark-job.yaml.template > k8s/benchmark-job.yaml

kubectl apply -f k8s/benchmark-job.yaml

echo "Job llm-benchmark-${JOB_SUFFIX} submitted."
echo "View logs with: kubectl logs -l job-name=llm-benchmark-${JOB_SUFFIX} -f"
