# Infinigram Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the infinigram search engine API.

## Overview

Infinigram is an engine that processes n-gram queries on massive text corpora with extremely low latency. This deployment includes:

- **API Server**: Flask-based REST API for n-gram queries
- **Persistent Storage**: PVC for storing index data
- **Load Balancing**: Kubernetes Service for internal load balancing
- **External Access**: Ingress for external access (optional)
- **Default Namespace**: All resources deployed in the default namespace

## Prerequisites

1. **Kubernetes Cluster**: A running Kubernetes cluster (minikube, kind, or cloud provider)
2. **kubectl**: Kubernetes command-line tool
3. **Docker**: For building the container image
4. **Nginx Ingress Controller**: For external access (optional)

### Installing Nginx Ingress Controller (Optional)

If you want external access via Ingress:

```bash
# For minikube
minikube addons enable ingress

# For other clusters
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
```

## Quick Start

1. **Deploy the application**:
   ```bash
   ./deploy.sh
   ```

2. **Check deployment status**:
   ```bash
   kubectl get pods -l app=infini-gram-api
   ```

3. **Access the API**:
   ```bash
   # Port forward to local machine
   kubectl port-forward service/infini-gram-service 8080:80
   
   # Test the API
   curl -X POST http://localhost:8080/ \
     -H "Content-Type: application/json" \
     -d '{
       "corpus": "your_index_name",
       "query_type": "count",
       "query": "example text"
     }'
   ```

## Architecture

### Components

1. **ConfigMap**: `infini-gram-config` - Configuration for the API server
2. **PersistentVolumeClaim**: `infini-gram-index-pvc` - Storage for index data
3. **Deployment**: `infini-gram-api` - Runs 3 replicas of the API server
4. **Service**: `infini-gram-service` - Internal load balancer
5. **Ingress**: `infini-gram-ingress` - External access (optional)
6. **HPA**: `infini-gram-hpa` - Horizontal Pod Autoscaler
7. **NetworkPolicy**: `infini-gram-network-policy` - Network security

### Resource Requirements

- **CPU**: 500m request, 1000m limit per pod
- **Memory**: 2Gi request, 4Gi limit per pod
- **Storage**: 100Gi for index data

## Configuration

### Environment Variables

The following environment variables can be configured via the ConfigMap:

- `FLASK_PORT`: API server port (default: 5000)
- `MODE`: Application mode (default: api)
- `MAX_QUERY_CHARS`: Maximum query characters (default: 1000)
- `MAX_QUERY_TOKENS`: Maximum query tokens (default: 500)
- `MAX_CLAUSES_PER_CNF`: Maximum clauses per CNF (default: 4)
- `MAX_TERMS_PER_CLAUSE`: Maximum terms per clause (default: 4)
- `MAX_SUPPORT`: Maximum support (default: 1000)
- `MAX_CLAUSE_FREQ`: Maximum clause frequency (default: 500000)
- `MAX_DIFF_TOKENS`: Maximum different tokens (default: 1000)
- `MAXNUM`: Maximum number (default: 10)
- `MAX_DISP_LEN`: Maximum display length (default: 10000)

### Index Configuration

The API server expects index data to be mounted at `/data/index`. You can:

1. **Pre-populate the PVC**: Copy your index files to the PVC before deployment
2. **Use an init container**: Add an init container to download/prepare index data
3. **Mount from external storage**: Use a different storage class or external volume

## API Usage

### Health Check

```bash
curl http://localhost:8080/health
```

### Query Examples

1. **Count query**:
   ```bash
   curl -X POST http://localhost:8080/ \
     -H "Content-Type: application/json" \
     -d '{
       "corpus": "your_index",
       "query_type": "count",
       "query": "example text"
     }'
   ```

2. **Probability query**:
   ```bash
   curl -X POST http://localhost:8080/ \
     -H "Content-Type: application/json" \
     -d '{
       "corpus": "your_index",
       "query_type": "prob",
       "query": "example text"
     }'
   ```

3. **Document search**:
   ```bash
   curl -X POST http://localhost:8080/ \
     -H "Content-Type: application/json" \
     -d '{
       "corpus": "your_index",
       "query_type": "search_docs",
       "query": "example text",
       "maxnum": 5
     }'
   ```

## Monitoring and Logs

### Check Pod Status

```bash
kubectl get pods -l app=infini-gram-api
```

### View Logs

```bash
# All pods
kubectl logs -l app=infini-gram-api -f

# Specific pod
kubectl logs <pod-name> -f
```

### Check Resource Usage

```bash
kubectl top pods -l app=infini-gram-api
```

## Scaling

### Horizontal Pod Autoscaler

Create an HPA for automatic scaling:

```bash
kubectl autoscale deployment infini-gram-api --cpu-percent=70 --min=3 --max=10
```

### Manual Scaling

```bash
kubectl scale deployment infini-gram-api --replicas=5
```

## Troubleshooting

### Common Issues

1. **Pods not starting**:
   ```bash
   kubectl describe pod <pod-name>
   ```

2. **Index not found**:
   - Ensure index data is properly mounted
   - Check the ConfigMap configuration

3. **Memory issues**:
   - Increase memory limits in deployment.yaml
   - Monitor resource usage

4. **Network issues**:
   - Check service configuration
   - Verify ingress controller is running

### Debug Commands

```bash
# Check events
kubectl get events --sort-by='.lastTimestamp'

# Check ConfigMap
kubectl get configmap infini-gram-config -o yaml

# Check PVC
kubectl get pvc infini-gram-index-pvc
```

## Cleanup

To remove the deployment:

```bash
./cleanup.sh
```

## Security Considerations

1. **Network Policies**: Consider adding network policies to restrict traffic
2. **RBAC**: Implement proper role-based access control
3. **Secrets**: Store sensitive configuration in Kubernetes secrets
4. **TLS**: Enable TLS for external access

## Performance Tuning

1. **Resource Limits**: Adjust CPU/memory based on your workload
2. **Replicas**: Scale based on expected load
3. **Storage**: Use SSD storage for better I/O performance
4. **Caching**: Consider adding Redis for caching frequently accessed data 