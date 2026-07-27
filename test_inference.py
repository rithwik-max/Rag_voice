from groq import Groq

def chatbot(prompt, groq_api_key):

    client = Groq(api_key=groq_api_key)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, knowledgeable and intelligent assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_completion_tokens=200
    )

    return response.choices[0].message.content