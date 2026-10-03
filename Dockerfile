# One container = the whole website: React app + FastAPI + trained model.
# Works on Render (sets $PORT automatically) and any Docker host (default port 7860).

# ---------- Stage 1: build the React app ----------
FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# The deployed frontend calls the API on the same site, under /api
ENV VITE_API_URL=/api
RUN npm run build

# ---------- Stage 2: Python API + model + website ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    TRAIN_N_JOBS=1 \
    PORT=7860

# Run as a normal (non-root) user
RUN useradd -m -u 1000 user
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY outputs/ outputs/
COPY --from=frontend /build/dist frontend/dist

# Train inside the image so the saved model always matches the installed
# library versions, then run the test suite. A failure stops the build.
WORKDIR /app/backend
RUN python -m app.ml.train && python -m pytest -q

RUN chown -R user:user /app
USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
