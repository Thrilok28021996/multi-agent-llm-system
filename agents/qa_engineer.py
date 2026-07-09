## QA Engineer

from openai import OpenAI


def qa_engineer(tasks):

    ## Prompt
    ## it should have model

    client = OpenAI(
        base_url="http://localhost:1234/v1/",
        api_key="lmstudio",  # required but ignored
    )

    response = client.responses.create(
        model="aisha/qwen3.5-4b-nothink",
        instructions="You are a QA Engineer. You will do the tasks assigned by the orchestrator.",
        input=tasks,
    )

    return response.output_text
