import os

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def get_pdf_text(pdf_documents):
    text = ""

    pdf_reader = PdfReader(pdf_documents)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def create_documents(pdf_files):
    documents = []

    for file in pdf_files:
        chunks = get_pdf_text(file)

        documents.append(
            Document(
                page_content=chunks,
                metadata={
                    "source": file,
                    "type": "PDF",
                    "owner": "Ramkumar JD"
                }
            ))

    return documents


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    return embeddings


def push_documents_to_pinecone(index_name, embeddings, documents):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    vector_store.add_documents(documents)


def main():
    try:
        load_dotenv()
        index_name = "intel-panam-index"
        directory_path = "./data/resumes"
        files = os.listdir(directory_path)

        pdf_files = []
        for file in files:
            pdf_file = directory_path + "/" + file
            pdf_files.append(pdf_file)
            print(f"Processing Initiated for the file {pdf_file}")

        documents = create_documents(pdf_files)
        embeddings = create_embeddings()
        push_documents_to_pinecone(
            index_name=index_name, embeddings=embeddings, documents=documents)

        print("Vector Embeddings are stored successfully into Pinecone Vector Database!")
    except Exception as e:
        print(f"Error Occurred, Details: {e}")
        raise


if __name__ == "__main__":
    main()
