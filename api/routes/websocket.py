import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Set
import logging
import os

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..dependencies import app_state
from ..model_manager import ModelManager

logger = logging.getLogger(__name__)

MAX_CLIENTS = int(os.environ.get("WS_MAX_CLIENTS", "100"))
MAX_BUFFER_SIZE = int(os.environ.get("WS_MAX_BUFFER_SIZE", "10000"))
MAX_BATCH_SAMPLES = int(os.environ.get("WS_MAX_BATCH_SAMPLES", "1000"))


router = APIRouter()


class ConnectionManager:
    def __init__(self, max_clients: int = MAX_CLIENTS):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_groups: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
        self.max_clients = max_clients

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        async with self._lock:
            if len(self.active_connections) >= self.max_clients:
                logger.warning(
                    f"Max clients ({self.max_clients}) reached. Rejecting {client_id}"
                )
                return False
            await websocket.accept()
            self.active_connections[client_id] = websocket
        logger.info(
            f"Client {client_id} connected. Total: {len(self.active_connections)}"
        )
        return True

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            for group in self.connection_groups.values():
                group.discard(client_id)
        logger.info(
            f"Client {client_id} disconnected. Total: {len(self.active_connections)}"
        )

    async def send_personal(self, client_id: str, message: dict) -> None:
        async with self._lock:
            ws = self.active_connections.get(client_id)
        if ws is not None:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast(
        self, message: dict, exclude: Optional[Set[str]] = None
    ) -> None:
        exclude = exclude or set()
        async with self._lock:
            snapshot = list(self.active_connections.items())
        for client_id, connection in snapshot:
            if client_id not in exclude:
                try:
                    await connection.send_json(message)
                except Exception:
                    await self.disconnect(client_id)

    async def broadcast_to_group(self, group_id: str, message: dict) -> None:
        if group_id not in self.connection_groups:
            return
        for client_id in self.connection_groups[group_id]:
            await self.send_personal(client_id, message)

    async def join_group(self, client_id: str, group_id: str) -> None:
        async with self._lock:
            if group_id not in self.connection_groups:
                self.connection_groups[group_id] = set()
            self.connection_groups[group_id].add(client_id)

    async def leave_group(self, client_id: str, group_id: str) -> None:
        async with self._lock:
            if group_id in self.connection_groups:
                self.connection_groups[group_id].discard(client_id)

    def get_connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()


@router.websocket("/predict")
async def websocket_predict(websocket: WebSocket, client_id: Optional[str] = None):
    if not client_id:
        client_id = f"client_{datetime.now(timezone.utc).timestamp()}"

    connected = await manager.connect(websocket, client_id)
    if not connected:
        await websocket.close(code=1013, reason="Max connections reached")
        return

    model_manager = app_state.model_manager
    if model_manager is None:
        await websocket.send_json(
            {
                "type": "error",
                "data": {"message": "Model manager not initialized"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        await manager.disconnect(client_id)
        return

    try:
        await websocket.send_json(
            {
                "type": "connected",
                "data": {
                    "client_id": client_id,
                    "message": "Connected to prediction stream",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "predict")

            if message_type == "predict":
                await handle_prediction(websocket, data, model_manager)

            elif message_type == "batch_predict":
                await handle_batch_prediction(websocket, data, model_manager)

            elif message_type == "ping":
                await websocket.send_json(
                    {
                        "type": "pong",
                        "data": {},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            elif message_type == "subscribe":
                group_id = data.get("data", {}).get("group_id")
                if group_id:
                    await manager.join_group(client_id, group_id)
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "data": {"group_id": group_id},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            elif message_type == "unsubscribe":
                group_id = data.get("data", {}).get("group_id")
                if group_id:
                    await manager.leave_group(client_id, group_id)
                    await websocket.send_json(
                        {
                            "type": "unsubscribed",
                            "data": {"group_id": group_id},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"message": f"Unknown message type: {message_type}"},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": str(e)},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception:
            pass
        await manager.disconnect(client_id)


async def handle_prediction(
    websocket: WebSocket, data: dict, model_manager: ModelManager
):
    try:
        payload = data.get("data", {})
        features = np.array(payload.get("features", []))
        model_name = payload.get("model_name")
        sequence_id = payload.get("sequence_id")

        if len(features) == 0:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": "No features provided"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return

        result = model_manager.predict(
            model_name=model_name, features=features, return_probabilities=True
        )

        await websocket.send_json(
            {
                "type": "prediction",
                "data": {
                    "prediction": result["prediction"],
                    "class_name": result["class_name"],
                    "confidence": result["confidence"],
                    "probabilities": result.get("probabilities", {}),
                    "model_used": result["model_used"],
                    "inference_time_ms": result["inference_time_ms"],
                    "sequence_id": sequence_id,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        await websocket.send_json(
            {
                "type": "error",
                "data": {"message": str(e)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


async def handle_batch_prediction(
    websocket: WebSocket, data: dict, model_manager: ModelManager
):
    try:
        payload = data.get("data", {})
        samples = payload.get("samples", [])
        model_name = payload.get("model_name")
        batch_id = payload.get("batch_id", datetime.now(timezone.utc).isoformat())

        if len(samples) == 0:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {"message": "No samples provided"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return

        if len(samples) > MAX_BATCH_SAMPLES:
            await websocket.send_json(
                {
                    "type": "error",
                    "data": {
                        "message": f"Batch size exceeds maximum ({MAX_BATCH_SAMPLES})"
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return

        await websocket.send_json(
            {
                "type": "batch_start",
                "data": {"batch_id": batch_id, "total_samples": len(samples)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        for idx, features in enumerate(samples):
            result = model_manager.predict(
                model_name=model_name,
                features=np.array(features),
                return_probabilities=True,
            )

            await websocket.send_json(
                {
                    "type": "batch_prediction",
                    "data": {
                        "batch_id": batch_id,
                        "sample_idx": idx,
                        "prediction": result["prediction"],
                        "class_name": result["class_name"],
                        "confidence": result["confidence"],
                        "probabilities": result.get("probabilities", {}),
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            await asyncio.sleep(0.01)  # prevent client overwhelm

        await websocket.send_json(
            {
                "type": "batch_complete",
                "data": {"batch_id": batch_id, "total_samples": len(samples)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        await websocket.send_json(
            {
                "type": "error",
                "data": {"message": str(e)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    client_id: Optional[str] = None,
    subject_id: Optional[str] = None,
):
    if not client_id:
        client_id = f"stream_{datetime.now(timezone.utc).timestamp()}"

    connected = await manager.connect(websocket, client_id)
    if not connected:
        return

    if subject_id:
        await manager.join_group(client_id, f"subject_{subject_id}")

    model_manager = app_state.model_manager

    try:
        await websocket.send_json(
            {
                "type": "stream_connected",
                "data": {
                    "client_id": client_id,
                    "subject_id": subject_id,
                    "message": "Connected to signal stream",
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        signal_buffer = {"ecg": [], "eda": [], "emg": [], "resp": [], "temp": []}
        window_size = 700

        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "signal")

            if message_type == "signal":
                signal_data = data.get("data", {})

                for signal_type in signal_buffer:
                    if signal_type in signal_data:
                        new_data = signal_data[signal_type]
                        if not isinstance(new_data, list):
                            continue
                        signal_buffer[signal_type].extend(new_data)
                        if len(signal_buffer[signal_type]) > MAX_BUFFER_SIZE:
                            signal_buffer[signal_type] = signal_buffer[signal_type][
                                -MAX_BUFFER_SIZE:
                            ]

                min_length = (
                    min(len(v) for v in signal_buffer.values() if len(v) > 0)
                    if any(len(v) > 0 for v in signal_buffer.values())
                    else 0
                )

                if min_length >= window_size:
                    features = []
                    for signal_type in ["ecg", "eda", "emg", "resp", "temp"]:
                        if signal_buffer[signal_type]:
                            signal = np.array(signal_buffer[signal_type][:window_size])
                            features.extend(
                                [
                                    np.mean(signal),
                                    np.std(signal),
                                    np.min(signal),
                                    np.max(signal),
                                ]
                            )

                    if features and model_manager:
                        result = model_manager.predict(
                            model_name=None,
                            features=np.array(features),
                            return_probabilities=True,
                        )

                        await websocket.send_json(
                            {
                                "type": "stream_prediction",
                                "data": {
                                    "prediction": result["prediction"],
                                    "class_name": result["class_name"],
                                    "confidence": result["confidence"],
                                    "probabilities": result.get("probabilities", {}),
                                    "window_timestamp": signal_data.get("timestamp"),
                                },
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )

                    # Slide window by half
                    for signal_type in signal_buffer:
                        if len(signal_buffer[signal_type]) > window_size:
                            signal_buffer[signal_type] = signal_buffer[signal_type][
                                window_size // 2 :
                            ]

            elif message_type == "config":
                config = data.get("data", {})
                if "window_size" in config:
                    requested = config["window_size"]
                    if (
                        isinstance(requested, int)
                        and 100 <= requested <= MAX_BUFFER_SIZE
                    ):
                        window_size = requested
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "data": {
                                    "message": f"window_size must be integer between 100 and {MAX_BUFFER_SIZE}"
                                },
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        continue
                await websocket.send_json(
                    {
                        "type": "config_updated",
                        "data": {"window_size": window_size},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            elif message_type == "reset":
                signal_buffer = {k: [] for k in signal_buffer}
                await websocket.send_json(
                    {
                        "type": "buffer_reset",
                        "data": {},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Stream error for {client_id}: {e}")
        await manager.disconnect(client_id)


@router.get("/connections")
async def get_connections():
    return {
        "active_connections": manager.get_connection_count(),
        "groups": {
            group_id: len(clients)
            for group_id, clients in manager.connection_groups.items()
        },
    }
