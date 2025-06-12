FROM python:3.12-slim-bookworm

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab stopwords wordnet

ENV NLTK_DATA=/usr/local/share/nltk_data

RUN adduser --system --group appuser

RUN chown -R appuser:appuser /usr/local/share/nltk_data
USER appuser

COPY ./app /app/app
COPY ./main.py /app/main.py

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]