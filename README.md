# tkt-text-embeddings

The text embeddings server behind the LLM Gateway, using Hugging Face TEI
(Text Embeddings Inference), with models loaded from the MLflow Model
Registry. It is the `text-embeddings` optional component of Thinkube.

## What it does

`server.py` is a FastAPI app on port 7860. It runs TEI
(`text-embeddings-router`) as a subprocess on port 8355 and passes requests
to it.

- **Model weights come from MLflow.** The server finds the latest version of
  the model in the MLflow Model Registry and gives TEI its path under
  `/mlflow-models`. Hugging Face runs offline (`HF_HUB_OFFLINE=1`).
- **It starts idle.** When `MODEL_ID` is set, it loads that model at start.
- **It switches models in place.** `POST /admin/switch-model` stops TEI,
  starts it with the new model, and rolls back if the new one does not
  start.
- **It gives an OpenAI-compatible API.** `POST /v1/embeddings` takes
  `input` (a string or a list) and returns OpenAI-style embeddings.
- **It has a test page.** A Gradio page at `/` shows the vector of one text,
  or the cosine similarity of two texts.

## How it reaches a user

It is the `text-embeddings` optional component of
[Thinkube](https://github.com/thinkube/thinkube). It is installed and
removed from the Optional Components page in thinkube-control. The
Templates page refuses it. The install deploys it with no pods
(`replicas: 0`, `gateway_managed: true` in `thinkube.yaml`); the LLM Gateway
creates a pod on a node when a model is loaded there. It is not installed on
its own.

## Overview

This template deploys an embedding server using:
- **TEI (Text Embeddings Inference)**: Hugging Face's Rust-based embedding server
- **MLflow Model Registry**: Models are downloaded from HuggingFace and stored in MLflow
- **OpenAI-compatible API**: `/v1/embeddings` endpoint works with any OpenAI SDK
- **ARM64 + CUDA**: the base image `text-embeddings-base` is built on the upstream TEI image for the node's GPU: `text-embeddings-inference:121-latest` (sm_121, DGX Spark) on arm64, `text-embeddings-inference:86-latest` (sm_86) on amd64

## Features

- OpenAI-compatible `/v1/embeddings` API endpoint
- Rust implementation (2-5x faster than Python)
- GPU acceleration (CUDA sm_121 on DGX Spark, sm_86 on amd64)
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
pod and registers the model. The docs page
[*Load a model on the node and context you choose*](https://github.com/thinkube/thinkube.org/blob/main/modules/ROOT/pages/playbooks/load-a-model-on-the-node-and-context-you-choose.adoc)
explains the steps.

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

These are the models the model catalogue
([thinkube-metadata `models.json`](https://github.com/thinkube/thinkube-metadata/blob/main/models.json))
marks for this component. A model that is not in the catalogue cannot be
mirrored or served.

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

### Multilingual
| Model | Parameters | Dimensions | Context | License |
|-------|------------|------------|---------|---------|
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 1024 | 8192 | Apache 2.0 |
| Qwen/Qwen3-Embedding-4B | 4B | 2560 | 8192 | Apache 2.0 |

## API Endpoints

On port 7860, the component's route:

- `POST /v1/embeddings` - Generate embeddings (OpenAI-compatible)
- `GET /v1/models` - The loaded model
- `GET /health` - Health check: `idle`, `switching`, `healthy`, or HTTP 503
- `GET /admin/current-model`, `POST /admin/switch-model`, `GET /admin/status` - Used by the LLM Gateway
- `GET /` - Gradio test page

TEI's own API (`/embed`, `/info`, `/metrics`) listens on port 8355 inside
the pod. The server calls its `/embed`.

## Architecture

```
[Client] --> [LLM Gateway] --> [server.py :7860] --> [TEI :8355] --> [Model from MLflow/JuiceFS]
```

Models are stored in MLflow Model Registry on JuiceFS shared storage, mounted at `/mlflow-models` in the container.

## Performance

TEI is built in Rust with optimized CUDA kernels, providing:
- 2-5x faster throughput than Python alternatives
- Automatic request batching
- Efficient GPU memory management
- Low latency (~5-10ms per batch)

## Working on it

| File | What it is |
|---|---|
| `server.py` | the FastAPI app, the TEI subprocess, the Gradio page |
| `entrypoint.sh` | sets offline Hugging Face mode and starts `server.py` |
| `thinkube_theme.py` | the Gradio theme |
| `Containerfile` | the image, on `text-embeddings-base` |
| `requirements.txt` | Python packages |
| `thinkube.yaml` | the component deployment: one GPU, port 7860 |
| `manifest.yaml` | the template metadata |

## License

MIT. Code generated from this template is yours: no attribution required, and you may license the app you build however you choose. See [LICENSE](LICENSE).

Copyright Alejandro Martínez Corriá and the Thinkube contributors
