import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import BingSearchAPIWrapper


@tool("Intermediate Answer")
def search(search_query: str):
    """
        Useful for when you need to search the internet for more information

        Args:
            search_query (str): search query
    """

    search = BingSearchAPIWrapper()

    return search.run(search_query)


def main():
    load_dotenv()

    tools = [search]

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4",
        max_tokens = 5000
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="self-ask-with-search",
        verbose=True,
        handle_parsing_errors=True,
        max_iterations = 5
    )

    prompt = "who was the president of USA when the sputnik satellite was launched"

    response = agent.invoke(prompt)

    print(response)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
