from pydantic import BaseModel


class Baby(BaseModel):
    gender: str