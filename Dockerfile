# --- Stage 1: Build Stage ---
FROM python:3.12-slim-bookworm as builder

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt stopwords wordnet


# --- Stage 2: Production Stage ---
FROM python:3.12-slim-bookworm

ENV NLTK_DATA=/usr/local/share/nltk_data

RUN adduser --system --group appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder --chown=appuser:appuser ${NLTK_DATA} ${NLTK_DATA}

COPY --from=builder --chown=appuser:appuser /app/app /app/app
COPY --from=builder --chown=appuser:appuser /app/main.py /app/main.py

USER appuser

EXPOSE 8080

CMD ["gunicorn", "main:app", "--workers", "9", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--timeout", "120"]