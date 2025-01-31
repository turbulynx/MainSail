```sql
docker volume create chroma_data
```

```sql
docker run -p 8000:8000 \
  -v chroma_data:/chroma/chroma.db \
  ghcr.io/chroma-core/chroma:latest \
  --host 0.0.0.0 --port 8000
```