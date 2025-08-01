#!/bin/bash

# Cleanup script for infinigram Kubernetes deployment
set -e

echo "🧹 Cleaning up infinigram deployment..."

# Delete all resources in the default namespace
kubectl delete deployment infinigram-api --ignore-not-found=true
kubectl delete service infinigram-service --ignore-not-found=true
kubectl delete ingress infinigram-ingress --ignore-not-found=true
kubectl delete configmap infinigram-config --ignore-not-found=true
kubectl delete pvc infinigram-index-pvc --ignore-not-found=true
kubectl delete hpa infinigram-hpa --ignore-not-found=true
kubectl delete networkpolicy infinigram-network-policy --ignore-not-found=true

# Remove Docker image (optional)
read -p "Do you want to remove the Docker image? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rmi infinigram-api:latest --force 2>/dev/null || true
    echo "🗑️  Docker image removed"
fi

echo "✅ Cleanup completed!" 