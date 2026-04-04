from dotenv import load_dotenv
import os

load_dotenv()

print("API key loaded:", bool(os.getenv("TINKER_API_KEY")))