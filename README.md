# tkt-text-embeddings

High-performance text embeddings server using Hugging Face TEI (Text Embeddings Inference), loading models from MLflow Model Registry.

## Overview

This template deploys an embedding server using:
- **TEI (Text Embeddings Inference)**: Hugging Face's optimized Rust-based embedding server
- **MLflow Model Registry**: Models are downloaded from HuggingFace and stored in MLflow
- **OpenAI-compatible API**: `/v1/embeddings` endpoint works with any OpenAI SDK
- **ARM64 + CUDA**: Built from source for DGX Spark (Blackwell GB10)

## Features

- OpenAI-compatible `/v1/embeddings` API endpoint
- High-performance Rust implementation (2-5x faster than Python)
- GPU acceleration (CUDA Blackwell sm_121a)
- Automatic batching and request queuing
- Models loaded from MLflow (no internet required at runtime)

## Usage

This repository is not deployed from the Templates page. It is the
`text-embeddings` optional component. The TEI base image it runs on is
built by the Thinkube installer.

### 1. Install the component

In thinkube-control, open *Optional Components* and install *Text
Embeddings*. This deploys the component with no pods.

### 2. Download a model

In thinkube-control, open the model catalogue, select an embedding model
(for example `nomic-ai/nomic-embed-text-v1.5`) and download it. It is
mirrored once into MLflow.

### 3. Load the model

Load the model on the node you choose. The LLM Gateway creates the serving
pod and registers the model. The docs page *Load a model on the node and
context you choose* explains the steps.

### 4. Use the API

The model is served through the LLM Gateway at `llm.<your-domain>`, with an
OpenAI-compatible embeddings endpoint:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llm.<your-domain>/v1",
    api_key="<your Thinkube API token>"
)

response = client.embeddings.create(
    model="nomic-ai/nomic-embed-text-v1.5",
    input=["Hello, world!", "This is a test."]
)

for embedding in response.data:
    print(f"Embedding dimension: {len(embedding.embedding)}")
```

## Supported Models

### General Purpose
| Model | Size | Dimensions | Context | License |
|-------|------|------------|---------|---------|
| nomic-ai/nomic-embed-text-v1.5 | ~550MB | 768 | 8192 | Apache 2.0 |
| BAAI/bge-base-en-v1.5 | ~440MB | 768 | 512 | MIT |
| BAAI/bge-large-en-v1.5 | ~1.3GB | 1024 | 512 | MIT |
| Alibaba-NLP/gte-large-en-v1.5 | ~1.6GB | 1024 | 8192 | Apache 2.0 |

### Code-Specific
| Model | Size | Dimensions | Context | License |
|-------|------|------------|---------|---------|
| jinaai/jina-embeddings-v2-base-code | ~560MB | 768 | 8192 | Apache 2.0 |
| jinaai/jina-embeddings-v3 | ~2.3GB | 1024 | 8192 | Apache 2.0 |

## API Endpoints

- `POST /v1/embeddings` - Generate embeddings (OpenAI-compatible)
- `POST /embed` - Simple embedding endpoint
- `GET /health` - Health check
- `GET /info` - Model information
- `GET /metrics` - Prometheus metrics

## Architecture

```
[Client] --> [TEI Server :8355] --> [Model from MLflow/JuiceFS]
```

Models are stored in MLflow Model Registry on JuiceFS shared storage, mounted at `/mlflow-models` in the container.

## Performance

TEI is built in Rust with optimized CUDA kernels, providing:
- 2-5x faster throughput than Python alternatives
- Automatic request batching
- Efficient GPU memory management
- Low latency (~5-10ms per batch)

## License

MIT. Code generated from this template is yours: no attribution required, and you may license the app you build however you choose. See [LICENSE](LICENSE).
