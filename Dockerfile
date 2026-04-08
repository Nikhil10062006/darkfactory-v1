FROM python:3.11-slim

# Designed for lightweight execution (vcpu=2, memory=8gb)
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY openenv.yaml .
COPY inference.py .

EXPOSE 7860

# HF Spaces expects the server to run indefinitely
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
