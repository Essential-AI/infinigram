#!/bin/bash

# Test script for infinigram Kubernetes deployment
set -e

echo "🧪 Testing infinigram deployment..."

# Wait for pods to be ready
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=infinigram --timeout=300s

# Check pod status
echo "📋 Pod status:"
kubectl get pods -l app=infinigram

# Test health endpoint
echo "🏥 Testing health endpoint..."
kubectl port-forward service/infinigram-service 8080:80 &
PF_PID=$!

# Wait for port forward to be ready
sleep 5

# Test health check
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    exit 1
fi

# Test API endpoint (basic test)
echo "🔍 Testing API endpoint..."
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "corpus": "test",
    "query_type": "count",
    "query": "test"
  }' || echo "⚠️  API test failed (expected if no index data)"

# Cleanup
kill $PF_PID 2>/dev/null || true

echo "✅ Deployment test completed!"
echo ""
echo "📊 To monitor the deployment:"
echo "   kubectl logs -l app=infinigram -f"
echo ""
echo "🌐 To access the API:"
echo "   kubectl port-forward service/infinigram-service 8080:80" 