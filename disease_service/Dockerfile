FROM python:3.10-slim

WORKDIR /app

COPY disease_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY disease_service /app/disease_service

EXPOSE 8010
CMD ["python", "-m", "disease_service.app"]
