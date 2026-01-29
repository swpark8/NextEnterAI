# import google.generativeai as genai
# Note: Keeping the above import commented out or removed.
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("⚠️ GOOGLE_API_KEY not found. Cannot configure Gemini.")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing models...")
try:
    # New SDK model listing might differ, usually client.models.list()
    # Iterate and print
    for m in client.models.list():
        print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")

