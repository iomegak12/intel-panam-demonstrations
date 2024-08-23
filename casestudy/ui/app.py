import os
import streamlit as st
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain


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


def get_summary_from_llm(llm, current_document):
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summary = chain.run([current_document])

    return summary


def main():
    try:
        load_dotenv()

        index_name = "intel-panam-index"
        model_name = "gpt-4"
        openai_api_key = os.environ["OPENAI_API_KEY"]

        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=4000,
            openai_api_key=openai_api_key
        )

        st.set_page_config(
            page_title="Resume Screening Assistant",
            page_icon=":sparkles:"
        )

        st.title("Resume Screening Assistant")
        st.subheader(
            "This AI Assistant would help you to screen available resumes, that are submitted to the Organization!"
        )

        job_description = st.text_area(
            "Please enter your Job Description Requirements ...",
            key="1",
            height=200
        )

        document_count = st.text_input(
            "No. Of Resume(s)",
            key="2"
        )

        submit = st.button("Analyze")

        if submit:
            embeddings = create_embeddings()
            relevant_documents = search_similar_documents(
                job_description,
                int(document_count),
                index_name=index_name,
                embeddings=embeddings
            )

            for document_index in range(len(relevant_documents)):
                st.subheader(":point_right: " + str(document_index+1))

                file_name = "*** FILE *** " + \
                    relevant_documents[document_index].metadata["source"]

                st.write(file_name)

                with st.expander("Show Me Summary of this Resume ..."):
                    summary = get_summary_from_llm(
                        llm=llm,
                        current_document=relevant_documents[document_index]
                    )

                    st.write("*** SUMMARY ***" + summary)
    except Exception as error:
        print(f"Error Occurred, Details: {error}")

        st.write("Error : {error}")

        raise


if __name__ == "__main__":
    main()
