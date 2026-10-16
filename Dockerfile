FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Install Python deps first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Chromium AND the OS libraries it needs. Because this runs after
# pip install, the browser build always matches the installed Playwright
# version, so there is no image-tag/version mismatch to manage.
RUN playwright install --with-deps chromium

COPY . .

# Render supplies $PORT. The app imports as "backend.app", so run from the
# repo root (templates are loaded from the relative path backend/templates).
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-10000}"]