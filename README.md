```sql
docker volume create chroma_data
```

```sql
docker run -p 8000:8000 \
  -v chroma_data:/chroma/chroma.db \
  ghcr.io/chroma-core/chroma:latest \
  --host 0.0.0.0 --port 8000
```

conda create --name mainsail python=3.12 


        pdf_path="./data/Documentation.pdf"

            google_api_key=os.getenv("GOOGLE_API_KEY")