import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from starlette.concurrency import run_in_threadpool


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
