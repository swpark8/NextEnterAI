import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"API Key present: {bool(api_key)}")

try:
    client = genai.Client(api_key=api_key)
    print("Client initialized successfully")
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents='Hello'
    )
    print("Generation success:", response.text)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
