#!/usr/bin/env bash

set -e # Terminate script immediately if any execution command throws an unhandled exception

echo "🚀 [Minikube Orchestrator] Starting infrastructure profile initialization..."
minikube start --cpus=4 --memory=8192 --driver=docker

echo "🔄 [Minikube Orchestrator] Pointing terminal CLI to internal Minikube registry..."
eval $(minikube docker-env)

echo "📦 [Minikube Orchestrator] Compiling application secure image layers..."
# Execute context generation from workspace root containing requirements.txt
# minikube image build -t whisper-backend:local-dev -f ./app/Dockerfile .
docker build -t whisper-backend:local-dev -f ./app/Dockerfile .

echo "⚙️ [Minikube Orchestrator] Deploying configuration assets and runtime instances..."
kubectl apply -f k8s-manifests/configmap.yaml
kubectl apply -f k8s-manifests/secret.yaml
kubectl apply -f k8s-manifests/cache-deployment.yaml
kubectl apply -f k8s-manifests/llm-deployment.yaml
kubectl apply -f k8s-manifests/backend-deployment.yaml

# kubectl get pods -l app.kubernetes.io/part-of=whisper-mesh -w

echo "🔍 [Minikube Orchestrator] Tracking rollout status. Waiting for pods to stabilize..."
kubectl rollout status deployment/whisper-fastapi-backend --timeout=120s

echo "🌐 [Minikube Orchestrator] Exposing backend service interface tunnel..."
echo "👉 Connect to REST endpoints via HTTP and real-time streams via ws:// on the printed port below."
minikube service whisper-backend-service