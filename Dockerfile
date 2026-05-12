ARG CONTAINER_REGISTRY
ARG TARGETARCH

# Select TEI base image per architecture:
#   - amd64: sm_86 (Ampere: RTX 3090, A100, etc.)
#   - arm64: sm_121 (Blackwell: DGX Spark GB10)
FROM ${CONTAINER_REGISTRY}/library/text-embeddings-inference:86-latest AS base-amd64
FROM ${CONTAINER_REGISTRY}/library/text-embeddings-inference:121-latest AS base-arm64
FROM base-${TARGETARCH}

# Install Python and Gradio dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages --upgrade -r /app/requirements.txt

# Copy application files
COPY entrypoint.sh /app/entrypoint.sh
COPY server.py /app/server.py
COPY thinkube_theme.py /app/thinkube_theme.py
RUN chmod +x /app/entrypoint.sh

# Copy assets
RUN mkdir -p /app/icons
COPY tk_ai.png /app/icons/tk_ai.png
COPY tk_ai.svg /app/icons/tk_ai.svg

# Create non-root user (matching UID from base image)
RUN useradd -m -u 1001 tei || true
USER tei
WORKDIR /app

# Expose ports: 8355 for TEI API, 7860 for Gradio UI
EXPOSE 8355 7860

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
