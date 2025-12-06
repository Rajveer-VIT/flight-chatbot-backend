import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Your ASP.NET Backend URL → MUST INCLUDE /api
FLIGHT_API_BASE_URL = os.getenv("FLIGHT_API_URL").rstrip("/")

