import base64
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def encode_image(image_file):
    with open(image_file, "rb") as file:
        return base64.b64encode(
            file.read()).decode("utf-8")


def main():
    load_dotenv()

    model_name = "gpt-4-turbo"
    openai_api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model=model_name,
                     temperature=0,
                     openai_api_key=openai_api_key)
    image_file = "./USMortgageRate.png"
    base64_image = encode_image(image_file=image_file)

    message = [
        {
            "role": "system",
            "content": "You're a Helpful Assistant that responds in Markdown"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the Image as an alternative text"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64, {base64_image}"
                    }
                }
            ]
        }
    ]

    response_message = llm.invoke(message)

    print(response_message.content)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
