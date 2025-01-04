from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.text_extraction import extract_text_from_pdf
from utils.embedding import add_to_chromadb, query_embedding
import uvicorn 
import os 
from google.generativeai import configure, GenerativeModel
from pydantic import BaseModel
import logging
from chromadb import Client

os.environ['GOOGLE_API_KEY'] = 'PLEASE PASTE YOUR GEMINI-FLASH-1.5 API KEY HERE'
configure(api_key = os.getenv('GOOGLE_API_KEY'))
app = FastAPI()

logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow React frontend to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Allow all headers
)

#endpopint to upload a pdf
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:

        pdf_path = f"./uploads/{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        logging.info(f"File uploaded and saved at: {pdf_path}")
        #extract text from uploaded file
        extracted_text = extract_text_from_pdf(pdf_path)
        logging.info('Text extraction complete')

        #put this text into chromadb
        add_to_chromadb(extracted_text)
        logging.info('Text added to ChromaDB')

        return {"message":f"File '{file.filename}' processed and added to the database, go ahead and ask any question related to it :)"}
    except Exception as e:
        logging.error(f"Error processing uploaded file: {e}")
        return {"error": "Internal Server error"}, 500

#endpoint to query the db search the db and return a response
class QueryRequest(BaseModel):
    text: str


@app.post("/query")
async def query_database(request: QueryRequest):
    #qe query the db for input query by the user 
    try:

        query_text = request.text
        logging.info(f"Recieved query: {query_text}")  

        #query chromadb for most relevant passage
        relevant_passage = query_embedding(query_text)
        logging.info(f"Retrieved passage: {relevant_passage}")

        #Call chromadb to find relevant doc
        result = query_embedding(query_text)
        logging.info(f"ChromaDB response: {result}")

        #make prompt such that reply is user friendly

        prompt = f"""
        You are a helpful and knowledgeable assistant. Answer the question based on the reference passage below.
        
        QUESTION: {query_text}
        PASSAGE: {relevant_passage}
        
        If the passage is irrelevant to the question, provide your best educated guess based on the available information.
        """
        logging.info(f"Generated prompt: {prompt}")

        #get response using gemini
        model = GenerativeModel("gemini-1.5-flash-latest")
        gemini_response = model.generate_content([prompt])
        logging.info(f"Gemini reponse: {gemini_response}")

        response_content = gemini_response.text
        logging.info(f"Gemini response content : {response_content}")

        return {"response": response_content}
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return {"error": "Internal server error"}, 500
        

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 8000)
