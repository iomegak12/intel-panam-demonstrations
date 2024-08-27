import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory


def main():
    load_dotenv()

    model_name = "gpt-4-turbo"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model=model_name,
                     temperature=0,
                     openai_api_key=openai_api_key)

    template = """
    You're a Chatbot that is helpful.
    Your goal is to help the user.
    Take what the user is saying and make a joke out of it.

    {chat_history}
    Human: {human_input}
    Chatbot:
    """

    prompt = PromptTemplate(
        input_variables=["human_input", "chat_history"],
        template=template,
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    question1 = "is an Pear a fruit or vegetable?"

    response = chain.predict(human_input = question1)

    print(response)

    question2 = "what was one of the fruits i first asked you about?"

    response2 = chain.predict(human_input = question2)

    print(response2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
