# --- Stage 1: Build Stage
FROM python:3.12-slim-bookworm as builder

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt_tab stopwords wordnet

# --- Stage 2: Production Stage
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder ${NLTK_DATA} ${NLTK_DATA}

RUN adduser --system --group appuser \
    && chown -R appuser:appuser ${NLTK_DATA}

USER appuser

COPY --from=builder /app/app /app/app
COPY --from=builder /app/main.py /app/main.py

EXPOSE 8080

CMD ["gunicorn", "main:app", "--workers", "9", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "--timeout", "120"]