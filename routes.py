from fastapi import APIRouter
from dto.pinecone_dto import pineconeDTO
from src import RAG_openai, pinecone_db


router = APIRouter()


@router.post("/chat")
async def RagChat(data: pineconeDTO):

    User_message = data.userMessage
    
    index_name = "pinecone-knowledgebase"
    index_name = index_name.lower().replace("_","-").replace(" ","-").replace(":","-")
    
    pinecone_DB = pinecone_db.vectorDB(index_name)
    pc = pinecone_db.pinecone_client

    if index_name not in pc.list_indexes().names():
        print("create vector db")
        pinecone_DB.upload_embeddings()


    source_knowledge = pinecone_DB.handle_query(User_message)

    # get model response
    model = RAG_openai.AIResponse(User_message, source_knowledge)
    model_response = model.generateResponse()


    return {
        "model_response": model_response,
        "Retriced data" : source_knowledge
        }


    


    
