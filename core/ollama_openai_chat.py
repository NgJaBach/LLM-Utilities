from openai import OpenAI
import openai
import backoff

client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.BadRequestError, openai.InternalServerError),
    max_time=999, max_tries=9999
)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def openai_ask(user_prompt: str, 
               sys_prompt: str = "",
               llm_name="qwen2.5:7b", 
               max_token=5000,
               temperature=0.3,
               ) -> str:
    response = completions_with_backoff(
        model=llm_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_token,
        n=1,
        stop=None,
        temperature=temperature,
    ).choices[0].message.content
    return response