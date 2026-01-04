# MobilityOne Production Deployment Guide

## Arhitektura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INTERNET                                      │
│                              │                                          │
│                      ┌───────▼───────┐                                  │
│                      │   TRAEFIK     │  :80, :443, :8080                │
│                      │   (Reverse    │                                  │
│                      │    Proxy)     │                                  │
│                      └───────┬───────┘                                  │
│                              │                                          │
│              ┌───────────────┼───────────────┐                          │
│              │               │               │                          │
│       ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐                   │
│       │  BOT API    │ │ ADMIN API   │ │  WORKERS    │                   │
│       │  :8000      │ │ :8080       │ │  (queue)    │                   │
│       │  PUBLIC     │ │ VPN ONLY    │ │             │                   │
│       └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                   │
│              │               │               │                          │
│       ┌──────▼───────────────▼───────────────▼──────┐                   │
│       │              SHARED SERVICES                 │                   │
│       │         Redis │ PostgreSQL │ Cache          │                   │
│       └─────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Brzi start

### 1. Priprema environment varijabli

```bash
# Kopiraj example datoteke
cp .env.example .env.prod
cp .env.admin.example .env.admin

# Generiraj sigurne tokene
openssl rand -hex 32  # Za ADMIN_TOKEN_1
openssl rand -hex 32  # Za ADMIN_TOKEN_2
openssl rand -hex 16  # Za DB_PASSWORD
```

### 2. Uredi .env.prod

```bash
# Obavezne varijable
DB_PASSWORD=your-secure-password
REDIS_PASSWORD=your-redis-password
OPENAI_API_KEY=sk-...
META_API_TOKEN=...
```

### 3. Uredi .env.admin

```bash
# Admin tokeni (64 karaktera)
ADMIN_TOKEN_1=your-64-char-token-here
ADMIN_TOKEN_2=another-64-char-token

# Dozvoljene IP adrese za Admin API
ADMIN_ALLOWED_IPS=10.0.0.0/8,192.168.1.0/24
```

### 4. Pokreni produkciju

```bash
# Osnovni servisi
docker-compose -f docker-compose.prod.yml up -d

# S monitoringom (Prometheus + Grafana)
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

## Network izolacija

### Mreže

| Network | Opis | Servisi |
|---------|------|---------|
| `public_net` | Internet-facing | Traefik |
| `bot_net` | Bot servisi | API, Workers, Autoscaler |
| `admin_net` | Admin only (internal) | Admin API, Grafana, Prometheus |
| `db_net` | Database (internal) | PostgreSQL, Redis |

### Portovi

| Port | Servis | Pristup |
|------|--------|---------|
| 80 | HTTP redirect | Public |
| 443 | HTTPS (Bot API) | Public |
| 8080 | Admin API | VPN/Intranet only |

## Firewall konfiguracija

### iptables (Linux)

```bash
# Dozvoli samo internu mrežu na Admin port
iptables -A INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### ufw (Ubuntu)

```bash
# Admin port samo za VPN
ufw allow from 10.0.0.0/8 to any port 8080
ufw deny 8080
```

## Admin API korištenje

### Autentifikacija

```bash
# Header: X-Admin-Token
curl -H "X-Admin-Token: your-token" \
     https://admin.mobilityone.io/admin/hallucinations
```

### Endpoints

| Method | Endpoint | Opis |
|--------|----------|------|
| GET | `/health` | Health check |
| GET | `/admin/hallucinations` | Lista halucinacija |
| POST | `/admin/hallucinations/{id}/review` | Označi pregledanim |
| GET | `/admin/statistics` | Dashboard statistike |
| GET | `/admin/audit-log` | Audit trail |
| GET | `/admin/export/training-data` | Export za fine-tuning |

### Primjer: Review halucinacije

```bash
curl -X POST \
     -H "X-Admin-Token: your-token" \
     -H "Content-Type: application/json" \
     -d '{
       "correction": "Točan limit je 3000 EUR, ne 5000 EUR",
       "category": "wrong_data"
     }' \
     https://admin.mobilityone.io/admin/hallucinations/hal_123/review
```

## Monitoring

### Pristup Grafani

```
URL: https://grafana.mobilityone.io (port 8080)
User: admin
Password: (iz .env.admin GRAFANA_PASSWORD)
```

### Ključne metrike

- `hallucinations_reported` - Broj prijavljenih halucinacija
- `false_positives_skipped` - Broj false positive detekcija
- `total_errors` - Ukupne greške
- `correction_rate` - Postotak automatskih ispravaka

## Backup

### PostgreSQL

```bash
# Backup
docker exec mobility_postgres pg_dump -U appuser mobility_db > backup.sql

# Restore
cat backup.sql | docker exec -i mobility_postgres psql -U appuser mobility_db
```

### Redis

```bash
# Backup
docker exec mobility_redis redis-cli BGSAVE

# Copy RDB file
docker cp mobility_redis:/data/dump.rdb ./backup/
```

### Error Learning Cache

```bash
# Backup .cache direktorija
docker cp mobility_api:/app/.cache ./backup/cache/
```

## Troubleshooting

### Admin API nije dostupan

1. Provjeri firewall:
   ```bash
   iptables -L -n | grep 8080
   ```

2. Provjeri Traefik IP whitelist:
   ```bash
   docker logs mobility_traefik 2>&1 | grep "admin"
   ```

3. Provjeri token:
   ```bash
   curl -v -H "X-Admin-Token: test" http://localhost:8080/health
   ```

### Rate limit exceeded

```bash
# Provjeri audit log
curl -H "X-Admin-Token: your-token" \
     https://admin.mobilityone.io/admin/audit-log
```

### Halucinacije se ne bilježe

1. Provjeri logove:
   ```bash
   docker logs mobility_api 2>&1 | grep "Hallucination"
   ```

2. Provjeri cache permissions:
   ```bash
   docker exec mobility_api ls -la /app/.cache/
   ```

## Scaling

### Workers

```bash
# Povećaj broj workera
WORKER_REPLICAS=4 docker-compose -f docker-compose.prod.yml up -d worker
```

### API (horizontal)

Koristi Docker Swarm ili Kubernetes za horizontalno skaliranje.

## Security Checklist

- [ ] `.env.prod` nije u git-u
- [ ] `.env.admin` nije u git-u
- [ ] Admin tokeni su 64+ karaktera
- [ ] Admin port (8080) je blokiran na firewallu za javni internet
- [ ] VPN je konfiguriran za admin pristup
- [ ] SSL certifikati su aktivni (Let's Encrypt)
- [ ] IP whitelist je konfiguriran u Traefiku
- [ ] Audit log je aktivan
- [ ] Backup je automatiziran
