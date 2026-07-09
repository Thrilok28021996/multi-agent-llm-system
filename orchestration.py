### Orchestration - where the llm that is good at instruction following will call
### the respective agent accordingly


from openai import OpenAI
import argparse
from agents.developer import developer
from agents.qa_engineer import qa_engineer

parser = argparse.ArgumentParser(
    prog="Orchestrator",
    description="What the program does",
    epilog="Text at the bottom of help",
)
parser.add_argument("--query", "--q")
args = parser.parse_args()


client = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key="lmstudio",  # required but ignored
)


# response = client.responses.create(
#     model="aisha/qwen3.5-4b-nothink",
#     instructions="You are a orchestrator. You will call the relevant agents to do the job.",
#     input=args.query,
# )

AGENTS = {"Developer": developer, "QA Engineer": qa_engineer}


router = client.chat.completions.create(
    model="aisha/qwen3.5-4b-nothink",
    messages=[
        {
            "role": "system",
            "content": "You are a router. Given a user request, output ONLY the name of the best agent "
            f"from this list: {list(AGENTS.keys())}. No other text.",
        },
        {"role": "user", "content": args.query},
    ],
    temperature=0.1,
)

print("calling the Agent:", router.choices[0].message.content)

if router.choices[0].message.content == "Developer":
    developer(args.query, 3)
else:
    resp = qa_engineer(args.query)
    print(resp)
