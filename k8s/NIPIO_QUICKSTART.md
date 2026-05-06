# Expose Microservices Without Buying a Domain (Local Cluster Friendly)

If your ingress external IP is `localhost`, your cluster is local (Docker Desktop/kind/minikube).
In that case, `nip.io` is not useful for public access. Use `localhost` for local testing, and a tunnel for public URLs.

## 1) Install NGINX ingress controller (one-time)

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
kubectl -n ingress-nginx get pods
```

Wait until controller pods are `Running`.

## 2) Check ingress controller endpoint

```bash
kubectl -n ingress-nginx get svc ingress-nginx-controller
```

If `EXTERNAL-IP` is `localhost`, continue with this guide.

## 3) Apply your services + ingress

```bash
kubectl apply -f k8s/frontend-svc.yaml
kubectl apply -f k8s/agri-gateway.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/disease-service-deployment.yaml
kubectl apply -f k8s/ingress-nipio.yaml
```

## 4) Verify ingress routes

```bash
kubectl get ingress agri-ingress
kubectl describe ingress agri-ingress
```

## 5) Local test URLs (same machine)

Use `Host` headers because ingress rules are host-based:

```bash
curl -I http://localhost -H "Host: app.1.2.3.4.nip.io"
curl -I http://localhost -H "Host: backend.1.2.3.4.nip.io"
curl -I http://localhost -H "Host: disease.1.2.3.4.nip.io"
```

If you use a NodePort/controller that is not bound to `localhost:80`, port-forward first:

```bash
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8080:80
```

Then:

```bash
curl -I http://localhost:8080 -H "Host: app.1.2.3.4.nip.io"
```

## 6) Public URL without domain (recommended for demos)

### Option A: Cloudflare Tunnel

```bash
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8080:80
cloudflared tunnel --url http://localhost:8080
```

### Option B: ngrok

```bash
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8080:80
ngrok http 8080
```

The tunnel command prints a public HTTPS URL you can share.

## Notes

- Keep backend/disease services as `ClusterIP` when using ingress only.
- `k8s/ingress-nipio.yaml` can still be used locally because the host value is matched from HTTP `Host` header.
- For real `nip.io` hostnames over the internet, you need a real public IP (not `localhost`).
