FROM python:3.12-slim

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY models/ models/

RUN mkdir -p /app/logs && chmod 777 /app/logs && chown -R appuser:appuser /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health'); exit(0)" || exit 1

USER appuser

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
