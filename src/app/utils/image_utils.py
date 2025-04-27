import aiofiles
import cv2
from fastapi import UploadFile, Path
from starlette.concurrency import run_in_threadpool


async def save_uploaded_file(upload_file: UploadFile, user_dir: Path, name: str) -> str:
    """Save uploaded file to a specific user's directory."""
    file_path = user_dir / f"{name}.jpg"
    async with aiofiles.open(file_path, "wb") as buffer:
        content = await upload_file.read()
        await buffer.write(content)
    await compress_image(str(file_path))
    return str(file_path)


async def compress_image(image_path: str) -> None:
    """Compress the image to reduce memory usage."""
    await run_in_threadpool(_compress_image_sync, image_path)


def _compress_image_sync(image_path: str) -> None:
    """Synchronous function to compress the image."""
    image = cv2.imread(image_path)
    if image is not None:
        compressed = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, compressed, [cv2.IMWRITE_JPEG_QUALITY, 85])
