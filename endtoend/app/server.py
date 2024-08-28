from decouple import config
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes


app = FastAPI()

model_name = "gpt-3.5-turbo-0125"
model = ChatOpenAI(
    temperature=0,
    openai_api_key=config("OPENAI_API_KEY"),
    model=model_name
)

template = """
    Give me a summary about {topic} in a paragraph or less"
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

add_routes(app, chain, path="/openai")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
