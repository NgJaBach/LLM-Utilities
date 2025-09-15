from ollama import chat


def ollama_ask(user_prompt: str, 
               sys_prompt: str = "",
               llm_name="qwen2.5:7b", 
               max_token=5000,
               temperature=0.3,
               reasoning_level="low", # ['low', 'medium', 'high']
               ) -> str:
    response = chat(
        model='gpt-oss:20b', 
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=reasoning_level,
        
    )
    # print(response.message.thinking)
    return response.message.content