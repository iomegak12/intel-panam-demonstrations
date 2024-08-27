import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, initialize_agent


def main():
    load_dotenv()

    model_name = "gpt-4-turbo"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model=model_name,
                     temperature=0,
                     openai_api_key=openai_api_key)

    tools = load_tools(
        [
            "llm-math"
        ],
        llm=llm
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(
        agent="conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        memory=memory
    )

    output1 = agent.run("Add 7 to 9 and tell me the result?")

    print(output1)

    output2 = agent.run("add 5 to the result and square root the result")

    print(output2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
