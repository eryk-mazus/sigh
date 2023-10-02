import openai

SYSTEM_PROMPT = (
    "You are a general purpose assistant. "
    "Your goal is to have a nice conversation with the user."
    "Be extroverted, answer the questions about yourself "
    "and express your preference."
    "The user messages are generated by a transcription service, "
    "which may contain spelling and grammar mistakes."
)


def get_gpt_reponse(prompt: str, stream: bool = True) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stream=stream,
    )

    for chunk in response:
        chunk_message = chunk["choices"][0]["delta"].get("content", "")
        print(chunk_message, end="", flush=True)
