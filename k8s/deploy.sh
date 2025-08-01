#!/bin/bash

# Deploy infinigram to Kubernetes
set -e

echo "🚀 Deploying infinigram to Kubernetes..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t infinigram:latest .

# Apply Kubernetes manifests
echo "🔧 Applying Kubernetes manifests..."

# Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# Apply PersistentVolumeClaim
kubectl apply -f k8s/persistent-volume-claim.yaml

# Apply Deployment
kubectl apply -f k8s/deployment.yaml

# Apply Service
kubectl apply -f k8s/service.yaml

# Apply Ingress (optional - requires nginx-ingress controller)
kubectl apply -f k8s/ingress.yaml

# Apply HPA for automatic scaling
kubectl apply -f k8s/hpa.yaml

# Apply Network Policy for security
kubectl apply -f k8s/network-policy.yaml

echo "✅ Deployment completed!"
echo ""
echo "📋 Status:"
kubectl get pods -l app=infinigram
echo ""
echo "🌐 To access the service:"
echo "   kubectl port-forward service/infinigram-service 8080:80"
echo ""
echo "📊 To check logs:"
echo "   kubectl logs -l app=infinigram -f" 