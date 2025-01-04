import chromadb
from chromadb import EmbeddingFunction, Embeddings
from google.generativeai import embed_content
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

#initialize chromadb
chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name = "googlecardb")

#adds the extracted text to chromadb as an embedding
def add_to_chromadb(text):
    try:
        logging.info("Generating embedding for document.")
        embedding_response = embed_content(
            model = "models/text-embedding-004",
            content = [text],
            task_type = "retrieval_document"
        )
        if "embedding" in embedding_response:
            embedding = embedding_response["embedding"][0]
        else:
            raise ValueError("Embedding not found in response. Check the API response format.")

        logging.info(f"Adding document to ChromaDB with first five embedding: {embedding[:5]}")
        doc_id = str(len(db.get()["documents"])) # get unique doc id

        db.add(documents = [text],
             embeddings=[embedding],
             ids = [doc_id])

        logging.info(f"Text added to chromadb successfully with ID: {doc_id}")
    except Exception as e:
        logging.error(f"Error adding text to ChromaDB: {e}")


#query the db for the most relevant document based on the query
def query_embedding(query_text):

    try:
        logging.info(f"Generating embedding for query: {query_text}")

        embedding_response = embed_content(
            model = "models/text-embedding-004",
            content = [query_text],
            task_type = "retrieval_query"
        )
        logging.info(f"Embedding response: {embedding_response}")

        query_embedding = embedding_response["embedding"][0]
        logging.info(f"Querying ChromaDb with embeddings: {query_embedding}")

        result = db.query(query_embeddings=[query_embedding], n_results=1)
        logging.info(f"Query result: {result}")

        if result["documents"]:
            return result["documents"][0]  # return most relevant document
        else:
            return "No relevant results found."
    except Exception as e:
        logging.error(f"Error querying ChromaDB: {e}")
        return "An error occured during the query"
