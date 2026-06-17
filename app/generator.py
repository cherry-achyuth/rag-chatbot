import os

from groq import Groq
from dotenv import load_dotenv

from config import LLM_MODEL

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(query, context):

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content