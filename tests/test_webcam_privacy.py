import ast
from pathlib import Path

WEBCAM_CAPTURE_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "api" / "webcam_capture.py"
)
PERSISTENCE_APIS = {
    "cv2.imwrite",
    "cv2.VideoWriter",
    "numpy.save",
    "numpy.savez",
    "numpy.savez_compressed",
    "numpy.memmap",
}


def _find_raw_frame_persistence_calls(source: str) -> list[str]:
    tree = ast.parse(source)
    module_aliases: dict[str, str] = {}
    function_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in {"cv2", "numpy"}:
                    module_aliases[imported.asname or imported.name] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module in {"cv2", "numpy"}:
            for imported in node.names:
                if imported.name in {"imwrite", "VideoWriter", "save", "savez", "savez_compressed"}:
                    function_aliases[imported.asname or imported.name] = (
                        f"{node.module}.{imported.name}"
                    )

    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            call_name = function_aliases.get(function.id)
        elif isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
            module_name = module_aliases.get(function.value.id)
            call_name = f"{module_name}.{function.attr}" if module_name else None
        elif isinstance(function, ast.Attribute) and function.attr == "tofile":
            call_name = "numpy.ndarray.tofile"
        else:
            call_name = None
        if call_name in PERSISTENCE_APIS or call_name == "numpy.ndarray.tofile":
            calls.append(call_name)
    return calls


def test_webcam_capture_does_not_persist_raw_frames():
    source = WEBCAM_CAPTURE_SOURCE.read_text(encoding="utf-8")
    assert _find_raw_frame_persistence_calls(source) == []


def test_webcam_privacy_detector_resolves_common_aliases():
    source = """
import cv2 as cv
import numpy as np
from cv2 import imwrite
from numpy import save as persist

cv.imwrite("frame.png", frame)
cv.VideoWriter("frames.mp4", 0, 30, (10, 10))
np.save("frame.npy", frame)
imwrite("frame.png", frame)
persist("frame.npy", frame)
"""
    assert _find_raw_frame_persistence_calls(source) == [
        "cv2.imwrite",
        "cv2.VideoWriter",
        "numpy.save",
        "cv2.imwrite",
        "numpy.save",
    ]
