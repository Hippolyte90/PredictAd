
import os
from dotenv import load_dotenv
from openai import OpenAI

from dotenv import load_dotenv


load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
openai = OpenAI()

# We will use GPT-4o-mini for wraping 

def message_gpt4(prompt, system_message):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return completion.choices[0].message.content


def stream_gpt4(prompt, system_message):
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": prompt})
    stream = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )

    response = ""
    for chunk in stream:
        text = chunk.choices[0].delta.content or ""
        response += text
        yield text
