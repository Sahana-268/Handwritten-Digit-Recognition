from __future__ import annotations

import argparse
import base64
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
STATIC_ROOT = PROJECT_ROOT / "web"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from digit_recognizer.model import DigitCNN
from digit_recognizer.preprocess import preprocess_digit_pil, save_preprocessed_preview


class Predictor:
    def __init__(self, checkpoint_path: Path, device_name: str = "cpu") -> None:
        self.device = torch.device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = DigitCNN().to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, object]:
        tensor = preprocess_digit_pil(image).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
        values, indices = torch.topk(probabilities, k=3)

        preview_path = PROJECT_ROOT / "artifacts" / "latest_live_preprocessed.png"
        save_preprocessed_preview(tensor.cpu(), preview_path)

        return {
            "digit": int(indices[0].item()),
            "confidence": float(values[0].item()),
            "probabilities": [float(value) for value in probabilities.tolist()],
            "top": [
                {"digit": int(index.item()), "probability": float(value.item())}
                for value, index in zip(values, indices)
            ],
        }


def parse_data_url(data_url: str) -> Image.Image:
    marker = "base64,"
    if marker not in data_url:
        raise ValueError("Image must be a base64 data URL.")

    encoded = data_url.split(marker, 1)[1]
    image_bytes = base64.b64decode(encoded)
    return Image.open(BytesIO(image_bytes)).convert("RGBA")


def build_handler(predictor: Predictor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = "/" if parsed.path == "/" else parsed.path

            if route == "/":
                self._serve_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            elif route == "/style.css":
                self._serve_file(STATIC_ROOT / "style.css", "text/css; charset=utf-8")
            elif route == "/app.js":
                self._serve_file(STATIC_ROOT / "app.js", "application/javascript; charset=utf-8")
            elif route == "/health":
                self._json({"status": "ok"})
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/predict":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                image = parse_data_url(str(payload.get("image", "")))
                result = predictor.predict(image)
                self._json(result)
            except Exception as error:
                self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _serve_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live digit recognition web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "best_model.pt",
        help="Trained checkpoint path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    predictor = Predictor(args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(predictor))
    url = f"http://{args.host}:{args.port}"
    print(f"Live digit recognizer running at {url}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()

