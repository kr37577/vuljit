FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/vuljit

# System deps (kept minimal; extend if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY .env.example ./.env.example
COPY plan.md ./plan.md
COPY replication ./replication
COPY vuljit ./vuljit
COPY scripts ./scripts
COPY analysis ./analysis

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "vuljit"]
CMD ["--help"]
