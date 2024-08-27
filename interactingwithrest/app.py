import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import APIChain


def main():
    load_dotenv()

    model_name = "gpt-4-turbo"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model=model_name,
                     temperature=0,
                     openai_api_key=openai_api_key)

    api_documentation = """
        BASE URL: https://restcountries.com/

        API Documentation:

        The API endpoint /v3.1/name/{name} Used to find informatin about a country. All URL parameters are listed below:
            - name: Name of country - Ex: italy, france

        The API endpoint /v3.1/currency/{currency} Used to find information about a region. All URL parameters are listed below:
            - currency: 3 letter currency. Example: USD, COP

        Woo! This is my documentation
    """

    chain = APIChain.from_llm_and_api_docs(
        llm,
        api_documentation,
        verbose=True,
        limit_to_domains=None

    )

    question = "can you tell me information about France?"

    response = chain.run(question)

    print(response)

    question2 = "Can you tell me about the currency COP?"

    response2 = chain.run(question2)

    print(response2)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
