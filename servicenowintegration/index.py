import os
import requests
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup


def access_service_now_kb_articles():
    headers = {
        "Content-Type": "application/json"
    }

    SN_BASE_URI = os.environ["SERVICE_NOW_BASE_URL"]
    SN_USERNAME = os.environ["SERVICE_NOW_USER"]
    SN_PASSWORD = os.environ["SERVICE_NOW_PASSWORD"]
    credentials = (SN_USERNAME, SN_PASSWORD)
    SN_KB_ACCESS_URL = f"{SN_BASE_URI}?sysparm_limit=25"
    response = requests.get(
        SN_KB_ACCESS_URL, auth=credentials, headers=headers)

    status = response.status_code
    processed_articles = []

    if status == 200:
        print("Authentication with ServiceNow is Successful ...")

        output_json = response.json()
        articles = output_json["result"]

        for article in articles:
            processed_articles.append(article)
    else:
        raise ("Authentication Failed!")

    return processed_articles


def fetch_and_process_html_text(text, title):
    soup = BeautifulSoup(text, "html.parser")
    document = Document(
        page_content=soup.text,
        metadata={
            "source": title,
            "type": "HTML",
            "owner": "Ramkumar JD",
        }
    )

    return document


def create_documents(articles):
    documents = []

    for article in articles:
        chunks = fetch_and_process_html_text(
            text=article["text"],
            title=article["short_description"]
        )

        documents.append(chunks)

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
        index_name = os.environ["PINECONE_INDEX"]
        articles = access_service_now_kb_articles()

        print(
            f"Totally {len(articles)} Article(s) Found, Started Embeddings and Indexing ...")

        documents = create_documents(articles=articles)
        embeddings = create_embeddings()
        push_documents_to_pinecone(
            index_name=index_name, embeddings=embeddings, documents=documents)

        print("Vector Embeddings are stored successfully into Pinecone Vector Database!")
    except Exception as e:
        print(f"Error Occurred, Details: {e}")
        raise


if __name__ == "__main__":
    main()
