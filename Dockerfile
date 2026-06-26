
# ── Base image ────────────────────────────────────────────
# Using Python 3.11 slim to keep the image lightweight
FROM python:3.11-slim

# ── Set working directory ─────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────
# libgomp1  → required by XGBoost
# gcc       → required to compile some Python packages
# default-libmysqlclient-dev → required by pymysql
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (layer caching) ───────────────
# Docker caches this layer — if requirements.txt hasn't changed
# it won't reinstall packages on every build
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Download spaCy language model ────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Copy project files ────────────────────────────────────
COPY . .

# ── Create necessary directories ──────────────────────────
RUN mkdir -p data/raw models

# ── Expose Streamlit port ─────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────
# Docker will check every 30 seconds if the app is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Startup command ───────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]