FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CHROME_BIN=/usr/bin/chromium

WORKDIR /workspace/RAISE

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    chromium \
    chromium-driver \
    curl \
    git \
    unzip \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/RAISE/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /workspace/RAISE/requirements.txt

COPY docker/entrypoint.sh /workspace/RAISE/docker/entrypoint.sh
RUN chmod +x /workspace/RAISE/docker/entrypoint.sh

COPY . /workspace/RAISE

ENTRYPOINT ["/workspace/RAISE/docker/entrypoint.sh"]
CMD ["run_all"]
