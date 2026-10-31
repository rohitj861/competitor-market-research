FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Unicode fonts for the PDF export: DejaVu (body, ₹/€/smart quotes), Lohit (Hindi), WenQuanYi (Chinese)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       fonts-dejavu-core fonts-dejavu-extra fonts-lohit-deva fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py eval_dataset.json ./

# No credentials are baked in: .env is excluded by .dockerignore and injected at runtime.
# Run as a non-root user; reports/ is where "Save PDF to reports folder" writes.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/reports \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
