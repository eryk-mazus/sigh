import openai

SYSTEM_PROMPT = (
    "You are a general purpose assistant. "
    "Your goal is to have a nice conversation with the user."
    "Be extroverted, answer the questions about yourself "
    "and express your preference."
    "The user messages are generated by a transcription service, "
    "which may contain spelling and grammar mistakes."
)


def get_gpt_reponse(prompt: str) -> str:
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
    )

    return response.choices[0].message["content"]
