# MobilityOne WhatsApp Bot - Deployment Guide
> Version: 4.0 | Last Updated: 2026-01-09

## Quick Navigation
- [Za DevOps Tim (Kubernetes)](#za-devops-tim-kubernetes-deployment)
- [Docker Compose (Development)](#docker-compose-development)
- [Environment Variables](#environment-varijable)
- [Cache & Embeddings (KRITIČNO!)](#cache--embeddings-kritično)
- [Health Checks](#health-checks)
- [Troubleshooting](#troubleshooting)

---

# Za DevOps Tim: Kubernetes Deployment

## Pregled Servisa

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES CLUSTER                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   API Pod   │    │ Worker Pod  │    │ Admin Pod   │                │
│   │   (1-3x)    │    │   (2-10x)   │    │    (1x)     │                │
│   │  Port 8000  │    │  (no port)  │    │  Port 8080  │                │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                │
│          │                  │                  │                        │
│          └──────────────────┼──────────────────┘                        │
│                             │                                           │
│                    ┌────────▼────────┐                                  │
│                    │  Shared Volume  │  ← PersistentVolumeClaim         │
│                    │   /app/.cache   │    (KRITIČNO - embeddings!)      │
│                    │     ~50MB       │                                  │
│                    └────────┬────────┘                                  │
│                             │                                           │
│          ┌──────────────────┼──────────────────┐                        │
│          │                  │                  │                        │
│   ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐                │
│   │  PostgreSQL │    │    Redis    │    │ Azure OpenAI│                │
│   │   (1x)      │    │    (1x)     │    │  (External) │                │
│   └─────────────┘    └─────────────┘    └─────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Servisi i Portovi

| Servis | Container Port | Replicas | Scaling | Napomena |
|--------|---------------|----------|---------|----------|
| **api** | 8000 | 1-3 | Manual/HPA | WhatsApp webhook endpoint |
| **worker** | - (no port) | 2-10 | KEDA (Redis lag) | Background processing |
| **admin-api** | 8080 | 1 | None | Internal only! |
| **postgres** | 5432 | 1 | None | StatefulSet |
| **redis** | 6379 | 1 | None | StatefulSet |

## Container Images

```bash
# Build all images
docker build -t mobilityone/api:latest .
docker build -t mobilityone/worker:latest .
docker build -t mobilityone/admin-api:latest .
docker build -t mobilityone/migration:latest .

# Svi koriste ISTI Dockerfile, razlika je u CMD:
# - api:       CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# - worker:    CMD ["python", "worker.py"]
# - admin-api: CMD ["uvicorn", "admin_api:app", "--host", "0.0.0.0", "--port", "8080"]
# - migration: CMD ["alembic", "upgrade", "head"]
```

---

## KRITIČNO: Cache Volume (Embeddings)

### Zašto je ovo kritično?

```
┌────────────────────────────────────────────────────────────────────────┐
│  /app/.cache/tool_embeddings.json = 40.9 MB                            │
│                                                                        │
│  BEZ OVOG FILEA:                                                       │
│  - API startup: ~60-120 sekundi (generira embeddings via Azure API)    │
│  - Troši Azure OpenAI API calls (~950 embedding requests)              │
│                                                                        │
│  S OVIM FILEOM:                                                        │
│  - API startup: <5 sekundi                                             │
│  - Nema Azure API poziva                                               │
└────────────────────────────────────────────────────────────────────────┘
```

### Rješenje: PersistentVolumeClaim

```yaml
# k8s/pvc-cache.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mobility-cache-pvc
spec:
  accessModes:
    - ReadWriteMany  # Svi podovi mogu čitati/pisati
  resources:
    requests:
      storage: 100Mi  # 50MB potrebno, 100Mi za headroom
  storageClassName: standard  # Prilagodite vašem clusteru
```

### Mount u Deploymentu

```yaml
# U svakom deploymentu (api, worker, admin-api):
spec:
  containers:
    - name: api
      volumeMounts:
        - name: cache-volume
          mountPath: /app/.cache
  volumes:
    - name: cache-volume
      persistentVolumeClaim:
        claimName: mobility-cache-pvc
```

### Cache Files

| File | Veličina | Opis | Kritičan? |
|------|----------|------|-----------|
| `tool_embeddings.json` | 40.9 MB | Semantic search vektori | **DA** |
| `swagger_manifest.json` | ~400 B | Cache verzija | DA |
| `tool_metadata.json` | ~3 MB | Tool definicije | DA |
| `error_learning.json` | ~100 KB | Naučene greške | NE |
| `api_capabilities.json` | ~10 KB | API capabilities | NE |

---

## Environment Varijable

### Obavezne (REQUIRED)

```bash
# === AZURE OPENAI (LLM + Embeddings) ===
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# === DATABASE ===
DATABASE_URL=postgresql+asyncpg://bot_user:password@postgres:5432/mobility_db
BOT_DATABASE_URL=postgresql+asyncpg://bot_user:password@postgres:5432/mobility_db
ADMIN_DATABASE_URL=postgresql+asyncpg://admin_user:password@postgres:5432/mobility_db

# === REDIS ===
REDIS_URL=redis://redis:6379/0

# === INFOBIP (WhatsApp) ===
INFOBIP_BASE_URL=your-instance.api.infobip.com
INFOBIP_API_KEY=your-api-key
INFOBIP_SENDER_NUMBER=385xxxxxxxxx
INFOBIP_SECRET_KEY=webhook-signature-key

# === MOBILITY ONE BACKEND ===
MOBILITY_API_URL=https://your-instance.mobilityone.io/
MOBILITY_AUTH_URL=https://your-instance.mobilityone.io/sso/connect/token
MOBILITY_CLIENT_ID=your-client-id
MOBILITY_CLIENT_SECRET=your-client-secret
MOBILITY_TENANT_ID=your-tenant-uuid

# === SWAGGER SOURCES ===
SWAGGER_URL=https://your-instance.mobilityone.io/automation/swagger/v1.0.0/swagger.json
```

### Opcionalne

```bash
# === MONITORING ===
SENTRY_DSN=https://xxx@sentry.io/xxx
GRAFANA_PASSWORD=admin

# === ADMIN API ===
ADMIN_TOKEN_1=64-char-hex-token
ADMIN_TOKEN_1_USER=admin.username
ADMIN_ALLOWED_IPS=10.0.0.0/8,192.168.0.0/16

# === PERFORMANCE ===
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
REDIS_MAX_CONNECTIONS=50
```

---

## Health Checks

### API Service

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120  # Embeddings mogu trajati!
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

startupProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 10
  failureThreshold: 30  # 30 × 10s = 5 min za startup
```

### Worker Service

```yaml
livenessProbe:
  exec:
    command:
      - pgrep
      - -f
      - "python worker.py"
  initialDelaySeconds: 30
  periodSeconds: 30
  failureThreshold: 3
```

### Admin API

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 30
```

---

## Database Setup

### Dual-User Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│  bot_user (LIMITED)           │  admin_user (FULL)              │
├───────────────────────────────┼─────────────────────────────────┤
│  ✅ SELECT/INSERT/UPDATE      │  ✅ ALL PRIVILEGES              │
│  ✅ conversations, messages   │  ✅ Can CREATE/ALTER tables     │
│  ✅ user_mappings             │  ✅ audit_logs (read)           │
│  ❌ CANNOT create tables      │  ✅ hallucination_reports       │
│  ❌ audit_logs (no access)    │  ✅ Migrations (alembic)        │
└───────────────────────────────┴─────────────────────────────────┘
```

### Migration Flow

```bash
# 1. PostgreSQL container starts → runs init-db.sh
#    Creates: mobility_db, bot_user, admin_user

# 2. Migration container runs (uses ADMIN_DATABASE_URL)
kubectl run migration --image=mobilityone/migration:latest --restart=Never

# 3. API/Worker start (use BOT_DATABASE_URL)
```

---

## Autoscaling (KEDA)

Worker skaliranje bazirano na Redis queue lag (NE CPU!).

### Zašto KEDA umjesto HPA?

```
AI workload = I/O bound, NOT CPU bound!

Čekanje Azure OpenAI odgovora:
- CPU: ~5%
- Latency: 2-5 sekundi
- HPA NE BI SKALIRAO jer je CPU nizak!

KEDA skalira na Redis Stream LAG:
- Ako 10+ poruka čeka → dodaj worker
- Ako <5 poruka → smanji workere
```

### Instalacija

```bash
# Install KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace

# Apply ScaledObject
kubectl apply -f k8s/keda-autoscaler.yaml
```

---

## Network & Security

### Ingress Rules

```yaml
# API - Public (WhatsApp webhook)
- host: bot.yourdomain.com
  paths:
    - path: /webhook
      service: api-service
      port: 8000

# Admin API - Internal Only!
- host: admin.internal.yourdomain.com  # VPN/Internal DNS only!
  paths:
    - path: /
      service: admin-service
      port: 8080
```

### Network Policies (Recommended)

```yaml
# Admin API can only be accessed from internal namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: admin-api-internal-only
spec:
  podSelector:
    matchLabels:
      app: admin-api
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: internal
```

---

## Startup Sequence

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STARTUP ORDER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. PostgreSQL    ───────────────────────────────────────→  Ready      │
│     └── init-db.sh creates users & database                            │
│                                                                         │
│  2. Redis         ───────────────────────────────────────→  Ready      │
│                                                                         │
│  3. Migration     ───────────────────────────────────────→  Complete   │
│     └── alembic upgrade head (creates tables)                          │
│                                                                         │
│  4. API           ─────────┬─────────────────────────────→  Ready      │
│     └── If no cache:       │                                           │
│         └── Generates embeddings (60-120s)                             │
│         └── Saves to /app/.cache/tool_embeddings.json                  │
│                            │                                           │
│  5. Worker        ─────────┴─────────────────────────────→  Ready      │
│     └── Waits for API health check                                     │
│     └── Reads embeddings from shared cache                             │
│                                                                         │
│  6. Admin API     ───────────────────────────────────────→  Ready      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Resource Requests/Limits

```yaml
# API
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

# Worker
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "1000m"

# Admin API
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

---

## Backup Strategy

### PostgreSQL

```bash
# CronJob za daily backup
kubectl create cronjob pg-backup \
  --image=postgres:15-alpine \
  --schedule="0 3 * * *" \
  -- pg_dump -h postgres -U admin_user mobility_db > /backup/$(date +%Y%m%d).sql
```

### Cache (Embeddings)

```bash
# Backup embeddings (opcionalno - mogu se regenerirati)
kubectl cp api-pod:/app/.cache/tool_embeddings.json ./backup/
```

---

## Troubleshooting

### API Startup Spor (>2 min)

```bash
# Provjeri ima li embeddings cache
kubectl exec -it api-pod -- ls -la /app/.cache/

# Ako nema tool_embeddings.json → generira se (~60-120s)
# Rješenje: Osiguraj PersistentVolume
```

### Worker Ne Procesira Poruke

```bash
# Provjeri Redis stream
kubectl exec -it redis-pod -- redis-cli XINFO GROUPS whatsapp_stream_inbound

# Provjeri worker logove
kubectl logs -f worker-pod
```

### Database Connection Errors

```bash
# Provjeri database URL
kubectl exec -it api-pod -- env | grep DATABASE

# Test konekcija
kubectl exec -it api-pod -- python -c "from database import engine; print(engine)"
```

### Health Check Fails

```bash
# Manual health check
kubectl exec -it api-pod -- curl -f http://localhost:8000/health

# Provjeri logove
kubectl logs api-pod --tail=100
```

---

## Checklist za Production

### Pre-Deploy

- [ ] PersistentVolumeClaim za `/app/.cache` kreiran
- [ ] Secrets kreirani (DATABASE_URL, API keys, etc.)
- [ ] PostgreSQL StatefulSet running
- [ ] Redis StatefulSet running
- [ ] Migration job completed successfully

### Post-Deploy

- [ ] API health check passing
- [ ] Worker processing messages
- [ ] Embeddings loaded (startup <10s after first run)
- [ ] WhatsApp webhook registered
- [ ] KEDA ScaledObject active

### Security

- [ ] Admin API NOT exposed to internet
- [ ] bot_user has limited DB permissions
- [ ] Secrets stored in K8s Secrets (not ConfigMaps)
- [ ] Network policies applied
- [ ] TLS enabled on Ingress

---

## Kontakt

Za pitanja o aplikaciji:
- Dokumentacija: `/docs/` folder
- Architecture: `docs/ARCHITECTURE.md`
- API Integration: `docs/API_INTEGRATION.md`

---

# Docker Compose (Development)

Za lokalni development:

```bash
# Start all services
docker-compose up -d

# Start with admin API
docker-compose --profile admin up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f api worker

# Rebuild after code changes
docker-compose build api worker
docker-compose up -d
```

## Portovi (Development)

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| Admin API | 8088 | http://localhost:8088 |
| PostgreSQL | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| Grafana | 3000 | http://localhost:3000 |
| Prometheus | 9090 | http://localhost:9090 |
