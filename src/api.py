"""
SkyGuard AI — Real-time backend
Replays the historical featured+labeled dataset as a live stream over WebSocket,
scoring each reading with the trained Random Forest model and attaching a
SHAP-based explanation whenever a reading is flagged as anomalous.

Run with:  uvicorn src.api:app --reload
Then open http://localhost:8000/docs to see the REST endpoints,
or connect a WebSocket client to ws://localhost:8000/ws
"""

import asyncio
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.config import EXPLAINER_PATH, FEATURED_DATA_PATH, FEATURES_PATH, MODEL_PATH, THRESHOLD
from src.explain import explain_row

SECONDS_PER_READING = 1.0  # how fast the simulator "ticks" — tune for demo pacing

# ---------------------------------------------------------------------------
# Load model artifacts once, at process startup
# ---------------------------------------------------------------------------
print("Loading model artifacts...")
rf_model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)
explainer = joblib.load(EXPLAINER_PATH)

print("Loading streaming dataset...")
stream_df = pd.read_csv(FEATURED_DATA_PATH)
stream_df["date"] = pd.to_datetime(stream_df["date"])
stream_df = stream_df.sort_values("date").reset_index(drop=True)
print(f"Loaded {len(stream_df)} rows ready to replay")

connected_clients: set[WebSocket] = set()


def score_row(row: pd.Series) -> dict:
    """Run the model + SHAP on a single row, return a dashboard-ready message."""
    X_row = row[feature_cols].to_frame().T.astype(float)
    prob = float(rf_model.predict_proba(X_row)[0, 1])
    is_anomaly = prob >= THRESHOLD

    message = {
        "station_id": row["station_id"],
        "date": str(row["date"].date()),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "temperature": round(float(row["temp_avg"]), 1),
        "humidity": round(float(row["humidity_avg"]), 1),
        "pressure": round(float(row["pressure_avg"]), 1),
        "probability": round(prob, 3),
        "is_anomaly": bool(is_anomaly),
        "reasons": [],
    }

    if is_anomaly:
        message["reasons"] = explain_row(X_row, explainer, feature_cols)

    return message


async def broadcast(message: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for d in dead:
        connected_clients.discard(d)


async def stream_loop():
    """Replays the dataset forever, one row every SECONDS_PER_READING seconds."""
    while True:
        for _, row in stream_df.iterrows():
            message = score_row(row)
            await broadcast(message)
            await asyncio.sleep(SECONDS_PER_READING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(stream_loop())
    yield
    task.cancel()


app = FastAPI(title="SkyGuard AI", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "SkyGuard AI backend running", "rows_loaded": len(stream_df)}


@app.get("/stations")
def list_stations():
    stations = stream_df[["station_id", "lat", "lon"]].drop_duplicates()
    return stations.to_dict(orient="records")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        connected_clients.discard(websocket)