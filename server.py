#!/usr/bin/env python3
"""
Text Embeddings Server with Gradio UI and Admin Management Endpoints
Uses TEI (text-embeddings-router) as a subprocess for model switching support.
Mirrors the vLLM server pattern for managed backend integration.
"""

import os
import json
import time
import signal
import logging
import asyncio
import subprocess

import httpx
import requests as req_lib
import numpy as np
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from thinkube_theme import create_thinkube_theme, THINKUBE_CSS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = None
MODEL_PATH = None

TEI_BACKEND_URL = "http://127.0.0.1:8355"
TEI_PID_FILE = "/tmp/tei.pid"

app = FastAPI(title="Text Embeddings Server")

backend_start_time = None
is_switching = False
http_client = None
sync_http_client = None


def query_mlflow(model_id: str) -> str:
    """Query MLflow to get the JuiceFS artifact path for a model."""
    token_response = req_lib.post(
        os.environ['MLFLOW_KEYCLOAK_TOKEN_URL'],
        data={
            'grant_type': 'password',
            'client_id': os.environ['MLFLOW_KEYCLOAK_CLIENT_ID'],
            'client_secret': os.environ['MLFLOW_CLIENT_SECRET'],
            'username': os.environ['MLFLOW_AUTH_USERNAME'],
            'password': os.environ['MLFLOW_AUTH_PASSWORD'],
            'scope': 'openid'
        },
        verify=False,
        timeout=30
    )
    token_response.raise_for_status()
    access_token = token_response.json()['access_token']

    model_name = model_id.replace('/', '-')
    mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow.mlflow.svc.cluster.local:5000')

    response = req_lib.get(
        f"{mlflow_url}/api/2.0/mlflow/model-versions/search",
        params={'filter': f"name='{model_name}'"},
        headers={'Authorization': f'Bearer {access_token}'},
        verify=False,
        timeout=30
    )
    response.raise_for_status()

    versions = response.json().get('model_versions', [])
    if not versions:
        raise ValueError(f"Model {model_name} not found in MLflow registry")

    latest = max(versions, key=lambda v: int(v['version']))
    run_id = latest['run_id']

    run_response = req_lib.get(
        f"{mlflow_url}/api/2.0/mlflow/runs/get",
        params={'run_id': run_id},
        headers={'Authorization': f'Bearer {access_token}'},
        verify=False,
        timeout=30
    )
    run_response.raise_for_status()
    experiment_id = run_response.json()['run']['info']['experiment_id']

    model_path = f'/mlflow-models/artifacts/{experiment_id}/{run_id}/artifacts/model'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    return model_path


def stop_backend():
    """Stop the current TEI subprocess using the PID file."""
    try:
        with open(TEI_PID_FILE) as f:
            pid = int(f.read().strip())
        logger.info(f"Stopping TEI (PID {pid})...")
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                break
        else:
            logger.warning(f"Force-killing TEI (PID {pid})")
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
            except ProcessLookupError:
                pass
        logger.info("TEI stopped")
    except (FileNotFoundError, ValueError, ProcessLookupError):
        logger.info("No running TEI backend found")


def start_backend(model_path: str, model_id: str) -> subprocess.Popen:
    """Start TEI as a background subprocess."""
    cmd = [
        "text-embeddings-router",
        "--model-id", model_path,
        "--port", "8355",
        "--hostname", "0.0.0.0",
    ]

    logger.info(f"Starting TEI: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    with open(TEI_PID_FILE, 'w') as f:
        f.write(str(proc.pid))
    return proc


def wait_for_backend(timeout: int = 120) -> bool:
    """Wait for TEI to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = req_lib.get(f"{TEI_BACKEND_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except req_lib.ConnectionError:
            pass
        time.sleep(2)
    return False


def initialize():
    """Initialize HTTP clients."""
    global http_client, sync_http_client
    logger.info(f"Backend URL: {TEI_BACKEND_URL}")
    logger.info("Starting in idle mode - no model loaded")

    http_client = httpx.AsyncClient(
        base_url=TEI_BACKEND_URL,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    sync_http_client = httpx.Client(
        base_url=TEI_BACKEND_URL,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


# ============================================================================
# Auto-load from MODEL_ID env var
# ============================================================================

def _do_auto_load(model_id: str):
    global MODEL_ID, MODEL_PATH, backend_start_time, is_switching

    logger.info(f"Auto-loading model from MODEL_ID env var: {model_id}")
    is_switching = True

    try:
        model_path = query_mlflow(model_id)
        logger.info(f"Resolved model path: {model_path}")

        start_backend(model_path, model_id)

        if not wait_for_backend(timeout=120):
            logger.error(f"Auto-load failed: {model_id} did not become healthy")
            is_switching = False
            return

        MODEL_ID = model_id
        MODEL_PATH = model_path
        backend_start_time = time.time()
        is_switching = False
        logger.info(f"Auto-load complete: {model_id}")
    except Exception as e:
        is_switching = False
        logger.error(f"Auto-load failed: {e}", exc_info=True)


@app.on_event("startup")
async def startup_auto_load():
    env_model_id = os.environ.get("MODEL_ID")
    if env_model_id:
        asyncio.create_task(asyncio.to_thread(_do_auto_load, env_model_id))


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    if MODEL_ID is None and not is_switching:
        return {"status": "idle", "model": None, "engine": "tei"}
    if is_switching:
        return {"status": "switching", "model": MODEL_ID, "engine": "tei"}
    try:
        response = await http_client.get("/health")
        backend_healthy = response.status_code == 200
        if backend_healthy:
            return {"status": "healthy", "model": MODEL_ID, "model_path": MODEL_PATH, "engine": "tei"}
        return JSONResponse(status_code=503, content={"status": "unhealthy", "model": MODEL_ID, "engine": "tei"})
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "model": MODEL_ID, "engine": "tei"}
        )


# ============================================================================
# Admin Management Endpoints
# ============================================================================

@app.get("/admin/current-model")
async def admin_current_model():
    if MODEL_ID is None:
        return {"model_id": None, "model_path": None, "status": "idle", "engine": "tei", "uptime_seconds": 0}
    uptime = time.time() - backend_start_time if backend_start_time else 0
    status = "switching" if is_switching else "serving"
    if not is_switching:
        try:
            r = await http_client.get("/health")
            if r.status_code != 200:
                status = "unhealthy"
        except Exception:
            status = "unhealthy"
    return {
        "model_id": MODEL_ID,
        "model_path": MODEL_PATH,
        "status": status,
        "engine": "tei",
        "uptime_seconds": round(uptime, 1),
    }


@app.post("/admin/switch-model")
async def admin_switch_model(request: Request):
    global MODEL_ID, MODEL_PATH, backend_start_time, is_switching

    if is_switching:
        return JSONResponse(
            status_code=409,
            content={"error": "A model switch is already in progress"}
        )

    body = await request.json()
    new_model_id = body.get("model_id")
    if not new_model_id:
        return JSONResponse(status_code=400, content={"error": "model_id is required"})

    previous_model = MODEL_ID
    previous_path = MODEL_PATH
    switch_start = time.time()

    try:
        is_switching = True

        logger.info(f"Switching model: {previous_model} -> {new_model_id}")
        new_model_path = query_mlflow(new_model_id)
        logger.info(f"Found model at: {new_model_path}")

        stop_backend()

        MODEL_ID = new_model_id
        MODEL_PATH = new_model_path
        os.environ["MODEL_ID"] = new_model_id
        os.environ["MODEL_PATH"] = new_model_path

        start_backend(new_model_path, new_model_id)

        if not wait_for_backend(timeout=120):
            logger.error(f"Failed to start TEI for {new_model_id}, rolling back...")
            stop_backend()
            MODEL_ID = previous_model
            MODEL_PATH = previous_path
            if previous_model and previous_path:
                try:
                    os.environ["MODEL_ID"] = previous_model
                    os.environ["MODEL_PATH"] = previous_path
                    start_backend(previous_path, previous_model)
                    wait_for_backend(timeout=120)
                    logger.info("Rollback succeeded")
                except Exception as rollback_err:
                    logger.error(f"Rollback also failed: {rollback_err}")

            is_switching = False
            backend_start_time = time.time() if MODEL_ID else None
            return JSONResponse(status_code=500, content={
                "previous_model": previous_model,
                "current_model": MODEL_ID,
                "status": "serving",
                "error": f"Failed to load {new_model_id}: backend did not become healthy within timeout"
            })

        backend_start_time = time.time()
        is_switching = False
        switch_time = time.time() - switch_start

        logger.info(f"Model switch complete: {previous_model} -> {MODEL_ID} in {switch_time:.1f}s")

        return {
            "previous_model": previous_model,
            "current_model": MODEL_ID,
            "status": "serving",
            "switch_time_seconds": round(switch_time, 1),
        }

    except Exception as e:
        logger.error(f"Error during model switch: {e}", exc_info=True)
        is_switching = False
        return JSONResponse(status_code=500, content={
            "previous_model": previous_model,
            "current_model": MODEL_ID,
            "status": "error",
            "error": str(e)
        })


@app.get("/admin/status")
async def admin_status():
    if is_switching:
        return {"status": "switching", "ready": False}
    if MODEL_ID is None:
        return {"status": "idle", "ready": True}
    try:
        r = await http_client.get("/health")
        if r.status_code == 200:
            return {"status": "ready", "ready": True}
        return {"status": "unhealthy", "ready": False}
    except Exception:
        return {"status": "unreachable", "ready": False}


# ============================================================================
# OpenAI-Compatible API Endpoints
# ============================================================================

@app.post("/v1/embeddings")
async def openai_embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint - proxies to TEI backend."""
    if MODEL_ID is None:
        return JSONResponse(status_code=503, content={"error": {"message": "No model loaded", "type": "service_unavailable"}})
    try:
        body = await request.json()

        # OpenAI uses 'input', TEI uses 'inputs'
        inputs = body.get('input', body.get('inputs', []))
        if isinstance(inputs, str):
            inputs = [inputs]

        # Forward to TEI backend
        response = req_lib.post(
            f"{TEI_BACKEND_URL}/embed",
            json={"inputs": inputs},
            timeout=60
        )
        response.raise_for_status()
        embeddings = response.json()

        # Format as OpenAI response
        return JSONResponse({
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(embeddings)
            ],
            "model": MODEL_ID,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        })
    except req_lib.exceptions.RequestException as e:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"TEI backend error: {str(e)}", "type": "backend_error"}}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


@app.get("/v1/models")
async def openai_models():
    """OpenAI-compatible models endpoint."""
    if MODEL_ID is None:
        return JSONResponse({"object": "list", "data": []})
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "thinkube",
            "permission": [],
            "root": MODEL_ID,
            "parent": None
        }]
    })


# ============================================================================
# Gradio Embeddings UI
# ============================================================================

def get_embedding(text: str) -> tuple[str, str]:
    """Get embedding vector for a single text."""
    if MODEL_ID is None:
        return "No model loaded", ""
    try:
        response = sync_http_client.post("/embed", json={"inputs": text})
        response.raise_for_status()
        embedding = response.json()[0]
        vector_preview = f"[{', '.join(f'{x:.4f}' for x in embedding[:10])}... ]"
        stats = f"Dimension: {len(embedding)}\nMagnitude: {np.linalg.norm(embedding):.4f}"
        return vector_preview, stats
    except Exception as e:
        return f"Error: {str(e)}", ""


def compare_texts(text1: str, text2: str) -> tuple[str, str, str]:
    """Compare two texts and show similarity."""
    if MODEL_ID is None:
        return "No model loaded", "", ""
    try:
        response = sync_http_client.post("/embed", json={"inputs": [text1, text2]})
        response.raise_for_status()
        embeddings = response.json()
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        interpretation = (
            "Very similar" if similarity >= 0.9 else
            "Similar" if similarity >= 0.7 else
            "Somewhat similar" if similarity >= 0.5 else
            "Low similarity" if similarity >= 0.3 else
            "Very different"
        )
        return f"Cosine Similarity: {similarity:.4f}", interpretation, f"Dimension: {len(vec1)}"
    except Exception as e:
        return f"Error: {str(e)}", "", ""


thinkube_theme = create_thinkube_theme()

with gr.Blocks() as demo:
    gr.Markdown("# Text Embeddings")
    gr.Markdown(f"Model: **{MODEL_ID or 'none (idle)'}**")

    with gr.Tabs():
        with gr.TabItem("Single Text"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Enter text", placeholder="Type or paste text...", lines=3)
                    embed_btn = gr.Button("Get Embedding", variant="primary")
                with gr.Column():
                    vector_output = gr.Textbox(label="Embedding Vector (preview)", lines=2)
                    stats_output = gr.Textbox(label="Statistics", lines=2)
            embed_btn.click(get_embedding, inputs=[text_input], outputs=[vector_output, stats_output])

        with gr.TabItem("Compare Texts"):
            with gr.Row():
                text1 = gr.Textbox(label="Text 1", placeholder="First text...", lines=3)
                text2 = gr.Textbox(label="Text 2", placeholder="Second text...", lines=3)
            compare_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                similarity_output = gr.Textbox(label="Similarity Score")
                interpretation_output = gr.Textbox(label="Interpretation")
                dim_output = gr.Textbox(label="Info")
            compare_btn.click(compare_texts, inputs=[text1, text2],
                            outputs=[similarity_output, interpretation_output, dim_output])

app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    favicon_path="/app/icons/tk_ai.png",
    theme=thinkube_theme,
    css=THINKUBE_CSS
)

if __name__ == "__main__":
    initialize()
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
