from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from .infer import predict

app = FastAPI(title="EXACT 2026 Starter API")


@app.post("/predict")
def predict_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    return predict(payload)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

