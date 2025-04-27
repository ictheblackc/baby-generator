import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from src.app.api.routes import router


load_dotenv()


app = FastAPI()
app.include_router(router)


api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
