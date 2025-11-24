# TOPIC 2: Request and Response
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# request content is a dictionary with the following keys:
# - role: the role of the content (user, assistant, system)
# - parts: a list of parts, each part is a dictionary with the following keys:
#   - text: the text of the part

part_content = {
    "text": "Explain how AI works in a few words",
}

request_content = {
    "role": "user",
    "parts": [part_content],
}

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=request_content,
)

# response is a dictionary with the many keys:
# - candidates: a list of candidates,
#               each candidate is a dictionary with the following keys:
#   - content: a dictionary with the following keys:
#     - role: the role of the content (user, assistant, system)
#     - parts: a list of parts, each part is a dictionary with the following keys:
#       - text: the text of the part

candidate = response.candidates[0]
print("role: ", candidate.content.role)
print("text: ", candidate.content.parts[0].text)
