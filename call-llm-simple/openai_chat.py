from ..constants import *
from openai import OpenAI
import openai
import backoff
import re

# client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
# client = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.BadRequestError, openai.InternalServerError),
    max_time=999, max_tries=9999
)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def remove_reasoning(response_content: str) -> str:
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        return final_answer
    else:
        return response_content.strip()

def ask(user: str, 
        llm_name="gpt-oss:20b", 
        max_token=128000,
        temperature=0.3,
        ) -> str:
    response = completions_with_backoff(
        model=llm_name,
        messages=[
            {"role": "user", "content": user},
        ],
        max_tokens=max_token,
        n=1,
        stop=None,
        temperature=temperature,
        reasoning={"effort": "medium"},
    ).choices[0].message.content
    return remove_reasoning(response)