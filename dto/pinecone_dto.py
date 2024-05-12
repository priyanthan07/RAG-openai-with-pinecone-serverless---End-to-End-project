from pydantic import BaseModel

class pineconeDTO(BaseModel):
    userMessage: str

    class Config:
        extra = 'forbid'