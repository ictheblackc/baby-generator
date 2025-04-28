import uuid
from src.app.utils.openai_utils import create_openai_client
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form
from fastapi import Depends
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from src.app.schemas.baby import Baby
from src.app.services.face_service import describe_face, combine_face_descriptions
from src.app.services.image_service import generate_prompt, generate_child_image
from src.app.utils.image_utils import save_uploaded_file
from src.app.config import MEDIA_DIR


router  = APIRouter()


@router.post("/generate-baby")
async def generate_baby(
    father_image: UploadFile = File(...),
    mother_image: UploadFile = File(...),
    gender: str = Form(...),
    client: AsyncOpenAI = Depends(create_openai_client)
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
        image_url = await generate_child_image(client, prompt, user_dir)

        # Clean up
        # os.remove(father_path)
        # os.remove(mother_path)

        return {"image_url": image_url}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})