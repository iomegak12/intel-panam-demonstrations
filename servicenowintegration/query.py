import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    return embeddings


def search_similar_documents(query, no_of_documents, index_name, embeddings):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    similar_documents = vector_store.similarity_search(
        query, k=no_of_documents)

    return similar_documents


def main():
    try:
        load_dotenv()

        index_name = os.environ["PINECONE_INDEX"]
        embeddings = create_embeddings()
        no_of_documents = 2
        query = """
            Can you explain me how to write an email signature?
        """

        relevant_documents = search_similar_documents(query=query, no_of_documents=no_of_documents,
                                                      index_name=index_name, embeddings=embeddings)

        for doc_index in range(len(relevant_documents)):
            document = relevant_documents[doc_index]

            print(document.metadata["source"])
            print(document.metadata["owner"])
            print(document.page_content)
            print("***** END *****")
            print("\n")
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()
