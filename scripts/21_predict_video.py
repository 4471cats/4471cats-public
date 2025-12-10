"""
video prediction server thing
basically takes video frames via websocket, runs yolo, sends back bboxes
also has cli for local detection

usage:
    uv run python scripts/21_predict_video.py serve --model weights/best.pt
    uv run python scripts/21_predict_video.py detect -m weights/best.pt -s video.mp4
    uv run python scripts/21_predict_video.py download-weights
"""

import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from ultralytics.models import YOLO

# 12 cat breeds from oxford pets
BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
]

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

app = typer.Typer()


def get_weights(dest: Path) -> Path:
    """grab weights from kaggle"""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading to {dest}")

    result = subprocess.run(
        ["kaggle", "kernels", "output", "inogai/4471cats-train1", "-p", str(dest)],
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info(result.stdout)

    # find the pt file
    pts = list(dest.glob("**/*.pt"))
    if not pts:
        raise FileNotFoundError("no .pt files??")

    pt = pts[0]
    if pt.parent != dest:
        # move it up
        target = dest / pt.name
        pt.rename(target)
        pt = target

    logger.success(f"got {pt}")
    return pt


def load_model(path: Path) -> YOLO:
    if not path.exists():
        logger.error(f"model not found: {path}")
        raise FileNotFoundError(path)

    logger.info(f"loading {path}")
    m = YOLO(str(path))
    logger.success("loaded")
    return m


def run_inference(model: YOLO, frame, conf: float = 0.25):
    """run yolo on frame, return detections"""
    results = model.predict(source=frame, conf=conf, verbose=False)

    dets = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            c = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            name = model.names.get(
                cls, BREEDS[cls] if cls < len(BREEDS) else f"cls_{cls}"
            )
            dets.append(
                {
                    "class": name,
                    "class_id": cls,
                    "confidence": round(c, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
    return dets


def draw_boxes(frame, dets):
    """draw bboxes on frame"""
    import cv2

    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        lbl = f"{d['class']} {d['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (w, h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            frame, lbl, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    return frame


# cli stuff


@app.command()
def download_weights(
    dest: Annotated[Path, typer.Option("-d", "--dest")] = WEIGHTS_DIR,
):
    """download weights from kaggle"""
    get_weights(dest)


@app.command()
def detect(
    model_path: Annotated[Path, typer.Option("-m", "--model")] = WEIGHTS_DIR
    / "best.pt",
    source: Annotated[str, typer.Option("-s", "--source")] = "0",
    conf: Annotated[float, typer.Option("-c", "--conf")] = 0.25,
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    show: Annotated[bool, typer.Option("--show")] = False,
    save_json: Annotated[Path | None, typer.Option("--save-json")] = None,
):
    """run detection on video/webcam"""
    import cv2

    model = load_model(model_path)

    # check if cv2 has gui
    has_gui = hasattr(cv2, "imshow")
    if show and not has_gui:
        logger.warning("no gui support, --show disabled")
        show = False

    # default output if nothing specified
    if not output and not show and not save_json:
        stem = Path(source).stem if not source.isdigit() else "webcam"
        output = Path("output") / f"detected_{stem}.mp4"
        logger.info(f"saving to {output}")

    # open video
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not Path(source).exists():
            logger.error(f"not found: {source}")
            raise typer.Exit(1)
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("cant open video")
        raise typer.Exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    logger.info(f"{w}x{h} @ {fps}fps")

    writer = None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    all_dets = []
    n = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            n += 1
            dets = run_inference(model, frame, conf)

            if save_json:
                all_dets.append({"frame": n, "detections": dets})

            annotated = draw_boxes(frame.copy(), dets)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("detect", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if dets:
                logger.info(
                    f"frame {n}: {[f'{d["class"]} ({d["confidence"]:.2f})' for d in dets]}"
                )

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if save_json:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(all_dets, indent=2))
        logger.success(f"saved json to {save_json}")

    logger.success(f"done, {n} frames")


@app.command()
def serve(
    model_path: Annotated[Path, typer.Option("-m", "--model")] = WEIGHTS_DIR
    / "best.pt",
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("-p", "--port")] = 8000,
    conf: Annotated[float, typer.Option("-c", "--conf")] = 0.25,
):
    """start api server"""
    import uvicorn

    logger.info(f"starting on {host}:{port}")
    logger.info(f"model: {model_path}")

    api = make_app(model_path, conf)
    uvicorn.run(api, host=host, port=port)


# fastapi app


def make_app(model_path: Path = WEIGHTS_DIR / "best.pt", default_conf: float = 0.25):
    import cv2
    import numpy as np
    from fastapi import APIRouter, FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    model = load_model(model_path)

    app = FastAPI(title="cat detector")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # serve client.html at root
    client_html = Path(__file__).parent / "client.html"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return client_html.read_text()

    # api router
    api = APIRouter(prefix="/api")

    @api.get("/classes")
    async def classes():
        return {"classes": list(model.names.values()) if model.names else BREEDS}

    @api.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        conf: float = Query(default_conf, ge=0, le=1),
        return_image: bool = False,
    ):
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(status_code=400, content={"error": "bad image"})

        dets = run_inference(model, frame, conf)

        if return_image:
            annotated = draw_boxes(frame, dets)
            _, buf = cv2.imencode(".jpg", annotated)
            return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")

        return {"detections": dets, "count": len(dets)}

    @api.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket):
        await ws.accept()
        logger.info("ws connected")

        conf = default_conf

        try:
            while True:
                data = await ws.receive_text()

                # config msg?
                try:
                    cfg = json.loads(data)
                    if "conf" in cfg:
                        conf = cfg["conf"]
                        await ws.send_json({"status": "ok", "conf": conf})
                        continue
                except json.JSONDecodeError:
                    pass

                # its an image
                t0 = time.time()
                try:
                    img = base64.b64decode(data)
                    arr = np.frombuffer(img, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    dets = run_inference(model, frame, conf)
                    ms = (time.time() - t0) * 1000

                    await ws.send_json(
                        {
                            "detections": dets,
                            "count": len(dets),
                            "ms": round(ms, 1),
                        }
                    )
                except Exception as e:
                    logger.error(f"frame error: {e}")
                    await ws.send_json({"error": str(e)})

        except WebSocketDisconnect:
            logger.info("ws disconnected")

    # keep the old annotated endpoint for compat i guess
    @api.websocket("/ws/stream-annotated")
    async def ws_annotated(ws: WebSocket):
        await ws.accept()
        conf = default_conf

        try:
            while True:
                data = await ws.receive_text()

                try:
                    cfg = json.loads(data)
                    if "conf" in cfg:
                        conf = cfg["conf"]
                        await ws.send_json({"status": "ok", "conf": conf})
                        continue
                except json.JSONDecodeError:
                    pass

                t0 = time.time()
                try:
                    img = base64.b64decode(data)
                    arr = np.frombuffer(img, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        await ws.send_json({"error": "bad frame"})
                        continue

                    dets = run_inference(model, frame, conf)
                    annotated = draw_boxes(frame, dets)

                    _, buf = cv2.imencode(".jpg", annotated)
                    b64 = base64.b64encode(buf).decode()

                    await ws.send_json(
                        {
                            "image": b64,
                            "detections": dets,
                            "count": len(dets),
                            "ms": round((time.time() - t0) * 1000, 1),
                        }
                    )
                except Exception as e:
                    await ws.send_json({"error": str(e)})

        except WebSocketDisconnect:
            pass

    app.include_router(api)
    return app


if __name__ == "__main__":
    app()
