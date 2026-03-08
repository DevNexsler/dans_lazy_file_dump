FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
# Filter out Apple Silicon-only packages (mlx-*) that won't install on Linux
RUN grep -v '^mlx' requirements.txt > requirements.docker.txt && \
    pip install --no-cache-dir -r requirements.docker.txt && \
    rm requirements.docker.txt

COPY . .

VOLUME /data

EXPOSE 7788

CMD ["python", "server.py"]
