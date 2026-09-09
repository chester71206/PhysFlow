from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2 as _cv2  # type: ignore
except Exception:
    _cv2 = None


COLOR_BGR2RGB = "COLOR_BGR2RGB"
COLOR_RGB2BGR = "COLOR_RGB2BGR"
IMREAD_UNCHANGED = -1


def has_native_cv2() -> bool:
    return _cv2 is not None and hasattr(_cv2, "imread")


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def _restore_dtype(image: np.ndarray, original: np.ndarray) -> np.ndarray:
    if original.dtype == np.uint8:
        return image
    if np.issubdtype(original.dtype, np.floating):
        restored = image.astype(np.float32)
        if original.max() <= 1.0:
            restored = restored / 255.0
        return restored.astype(original.dtype, copy=False)
    return image.astype(original.dtype, copy=False)


def imread(path: str, flags: int | None = None) -> np.ndarray | None:
    if has_native_cv2():
        return _cv2.imread(path, flags)  # type: ignore[arg-type]

    with Image.open(path) as img:
        if flags == IMREAD_UNCHANGED:
            return np.array(img)
        rgb = np.array(img.convert("RGB"))
    return rgb[:, :, ::-1]


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if has_native_cv2():
        return _cv2.resize(image, size)

    original = image
    image_uint8 = _ensure_uint8(image)
    pil_image = Image.fromarray(image_uint8)
    resized = np.array(pil_image.resize(size, resample=Image.BILINEAR))
    return _restore_dtype(resized, original)


def cvtColor(image: np.ndarray, code: str) -> np.ndarray:
    if has_native_cv2():
        native_code = getattr(_cv2, code)
        return _cv2.cvtColor(image, native_code)

    if code in {COLOR_BGR2RGB, COLOR_RGB2BGR}:
        return image[:, :, ::-1]
    raise ValueError(f"Unsupported conversion code: {code}")


def imwrite(path: str, image: np.ndarray) -> bool:
    if has_native_cv2():
        return bool(_cv2.imwrite(path, image))

    array = _ensure_uint8(image)
    if array.ndim == 3 and array.shape[2] == 3:
        array = array[:, :, ::-1]
    Image.fromarray(array).save(path)
    return True
