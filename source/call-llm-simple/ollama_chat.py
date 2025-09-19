# https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

from ollama import chat

def ollama_ask(user_prompt: str, 
               sys_prompt: str = "",
               llm_name="qwen2.5:7b", 
               max_token=128000,
               temperature=0.3,
               reasoning_level=None, # ['low', 'medium', 'high']
               ) -> str:
    response = chat(
        model=llm_name, 
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=reasoning_level,
        options={"max_tokens": max_token, "temperature": temperature, }
    )
    # print(response.message.thinking)
    return response.message.content

if __name__ == "__main__":
    question = "What is the capital of France?"
    answer = ollama_ask(question)
    print("Q:", question)
    print("A:", answer)