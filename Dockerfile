FROM python:3.12-slim

WORKDIR /app

# Build tools needed for some wheels (faiss/sentence-transformers are prebuilt).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image build time so the app starts ready.
RUN python -m app.ingest

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
