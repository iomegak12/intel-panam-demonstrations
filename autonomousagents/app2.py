import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts.prompt import PromptTemplate


@tool
def multiply(first_int: int, second_int: int) -> int:
    """
        multiply two integers together
    """

    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """
        adds or sums two integers together
    """

    return first_int + second_int


@tool
def divide(first_int: int, second_int: int) -> int:
    """
        divide two integers 
    """

    return first_int / second_int


@tool
def subtract(first_int: int, second_int: int) -> int:
    """
        substracts two integers together
    """

    return first_int - second_int


@tool
def exponentize(base: int, exponent: int) -> int:
    """
        exponentize the base to the exponent value
    """

    return base ** exponent


def main():
    load_dotenv()

    model_name = "gpt-4-turbo"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model=model_name,
                     temperature=0,
                     openai_api_key=openai_api_key)

    tools = [add, multiply, subtract, divide, exponentize]
    template = """
        SYSTEM
        You are a helpful assistant

        PLACEHOLDER
        chat_history

        HUMAN
        {input}

        PLACEHOLDER
        {agent_scratchpad}
    """

    prompt = PromptTemplate(
        input_variables=["input", "chat_history"],
        template=template,
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    question = """
        Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result.
    """

    response = agent_executor.invoke({
        "input": question
    })

    print(response)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
