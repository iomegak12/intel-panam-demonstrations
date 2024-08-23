import os
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI


def main():
    load_dotenv()

    st.set_page_config(
        page_title="CSV Document - Analysis",
        page_icon=":books:"
    )

    st.title("HR - Attrition Analysis Chatbot")
    st.subheader("Helps to conver insights from HR Attrition Data!")

    st.markdown(
        """
            This chatbot is created to demonstrate and answer questions from a set of attributes data present in CSV file. And that file was curated
            by Organization Engineering Team to help LLMs to understand the data better.
        """
    )

    user_question = st.text_input(
        "Ask your question about HR Attritions Data ...")
    csv_path = "./hr-employees-attritions-internet.csv"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        max_tokens=1000,
        openai_api_key=openai_api_key)
    agent = create_csv_agent(
        llm,
        [csv_path],
        verbose=True,
        allow_dangerous_code=True,
    )

    agent.handle_parsing_errors = True

    answer = agent.invoke(user_question)

    st.write(answer["output"])


if __name__ == "__main__":
    main()
