FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/vuljit

# System deps (kept minimal; extend if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY vuljit/requirements.txt /app/vuljit/requirements.txt
RUN pip install --no-cache-dir -r /app/vuljit/requirements.txt

COPY vuljit /app/vuljit

ENTRYPOINT ["python", "-m", "vuljit"]
CMD ["--help"]

