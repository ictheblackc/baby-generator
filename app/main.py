import os
import shutil
import uuid
import aiofiles
from pathlib import Path

from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import face_recognition
import uvicorn
from deepface import DeepFace
from starlette.concurrency import run_in_threadpool


app = FastAPI()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)


MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)


class Baby(BaseModel):
    gender: str


@app.post("/generate-baby")
async def generate_baby(
    father_image: UploadFile = File(...),
    mother_image: UploadFile = File(...),
    gender: str = Form(...)
):
    try:
        # Create a developed directory for the user
        session_id = uuid.uuid4().hex
        user_dir = Path(MEDIA_DIR) / session_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Save images to temporary paths
        father_path = await save_uploaded_file(father_image, user_dir, "father")
        mother_path = await save_uploaded_file(mother_image, user_dir, "mother")

        # Extract face features
        father_desc = await describe_face(father_path)
        mother_desc = await describe_face(mother_path)

        # Merge attributes
        combined_desc = await combine_face_descriptions(father_desc, mother_desc)

        # Generate prompt and image
        prompt = await generate_prompt(combined_desc, gender)
        image_url = await generate_child_image(prompt)

        # Clean up
        # os.remove(father_path)
        # os.remove(mother_path)

        return {"image_url": image_url}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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


async def describe_face(image_path: str) -> dict:
    """Extract features using DeepFace and color/landmark analysis."""
    return await run_in_threadpool(_describe_face_sync, image_path)


def _describe_face_sync(image_path: str) -> dict:
    """Synchronous function to describe the face."""
    df = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=True)[0]

    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        raise ValueError("No face found in the image")

    top, right, bottom, left = face_locations[0]
    landmarks = face_recognition.face_landmarks(rgb)[0]

    # Eye color extraction
    eye_colors = []
    for eye_key in ['left_eye', 'right_eye']:
        eye = landmarks[eye_key]
        mask = np.zeros_like(rgb)
        cv2.fillPoly(mask, [np.array(eye)], (255, 255, 255))
        masked = cv2.bitwise_and(rgb, mask)
        pixels = masked[np.where((masked != [0, 0, 0]).all(axis=2))]
        if len(pixels) > 0:
            mean_color = np.mean(pixels, axis=0)
            eye_colors.append(tuple(mean_color.astype(int)))

    avg_eye_color = np.mean(eye_colors, axis=0).astype(int) if eye_colors else (0, 0, 0)

    # Hair color estimation
    hair_region = rgb[top:top + (bottom - top) // 4, left:right]
    hair_color = np.mean(hair_region.reshape(-1, 3), axis=0).astype(int) if hair_region.size > 0 else (0, 0, 0)

    return {
        "age": df.get("age"),
        "gender": df.get("dominant_gender"),
        "race": df.get("dominant_race"),
        "emotion": df.get("dominant_emotion"),
        "eye_color_rgb": tuple(avg_eye_color),
        "hair_color_rgb": tuple(hair_color),
        "landmarks": landmarks
    }


async def combine_face_descriptions(father: dict, mother: dict) -> dict:
    """Average relevant attributes of both parents."""
    return {
        "race": mother["race"],  # Prefer mother's skin tone
        "eye_color_rgb": tuple(((np.array(father["eye_color_rgb"]) + np.array(mother["eye_color_rgb"])) / 2).astype(int)),
        "hair_color_rgb": tuple(((np.array(father["hair_color_rgb"]) + np.array(mother["hair_color_rgb"])) / 2).astype(int)),
        "emotion": father["emotion"],  # Optionally choose dominant emotion
    }


async def generate_prompt(desc: dict, gender: str) -> str:
    """Build the prompt for the AI image generation based on attributes."""
    return (
        f"A photorealistic studio portrait of a {gender} child. "
        f"The child has skin tone similar to the {desc['race']}, "
        f"eyes approximately the color {desc['eye_color_rgb']}, "
        f"hair color around {desc['hair_color_rgb']}, "
        f"and a facial expression resembling {desc['emotion']}. "
        f"Studio lighting, high resolution, realistic style."
    )


async def generate_child_image(prompt: str) -> str:
    """Generate image using OpenAI DALL-E model."""
    response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
