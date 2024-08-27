import requests
import os
from dotenv import load_dotenv


def main():
    load_dotenv()

    headers = {
        "Content-Type": "application/json"
    }

    SN_BASE_URI = os.environ["SERVICE_NOW_BASE_URL"]
    SN_USERNAME = os.environ["SERVICE_NOW_USER"]
    SN_PASSWORD = os.environ["SERVICE_NOW_PASSWORD"]
    credentials = (SN_USERNAME, SN_PASSWORD)
    SN_KB_ACCESS_URL = f"{SN_BASE_URI}?sysparm_limit=25"
    response = requests.get(
        SN_KB_ACCESS_URL, auth=credentials, headers=headers)

    status = response.status_code

    if status == 200:
        print("Authentication with ServiceNow is Successful ...")

        output_json = response.json()
        response_text = []

        for article in output_json["result"]:
            description = article["short_description"]

            print(f"{description}")

            response_text.append(article["text"])

        no_of_results = len(response_text)
        print(f"Totally {no_of_results} Article(s) Found!")
    else:
        raise ("Authentication Failed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")
