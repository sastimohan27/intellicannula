"""Real-time alert WebSocket.

Kept deliberately simple: an in-process set of connected sockets. That is
correct for a single-worker deployment and is what the prototype runs. With
multiple workers each would hold its own set and a clinician connected to
worker B would miss an alert raised on worker A - so a multi-worker deployment
needs a shared broker (Redis pub/sub). Noted in docs/deployment_checklist.md
rather than pretended away.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas import AlertMessage

log = logging.getLogger(__name__)
router = APIRouter(tags=["alerts"])

_connections: set[WebSocket] = set()


@router.websocket("/ws/alerts")
async def alerts_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    log.info("alert socket connected (%d total)", len(_connections))
    try:
        while True:
            # The client sends nothing meaningful; this keeps the socket open
            # and gives us a clean disconnect signal.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _connections.discard(websocket)
        log.info("alert socket disconnected (%d remain)", len(_connections))


async def broadcast_alert(message: AlertMessage) -> int:
    """Push an alert to every connected client.

    A failed send must not abort the others or fail the request that raised the
    alert - the check-in is already persisted, and a dropped socket is a client
    problem, not a clinical one. Dead sockets are pruned as they are found.
    """
    if not _connections:
        return 0

    payload = message.model_dump(mode="json")
    delivered, dead = 0, []
    for socket in list(_connections):
        try:
            await socket.send_json(payload)
            delivered += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("dropping alert socket: %s", exc)
            dead.append(socket)

    for socket in dead:
        _connections.discard(socket)
    return delivered


def connection_count() -> int:
    return len(_connections)
