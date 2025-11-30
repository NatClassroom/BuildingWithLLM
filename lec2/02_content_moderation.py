from google import genai
from pydantic import BaseModel, Field
from typing import Union, Literal
import os
from dotenv import load_dotenv

load_dotenv()


class SpamDetails(BaseModel):
    reason: str = Field(description="The reason why the content is considered spam.")
    spam_type: Literal["phishing", "scam", "unsolicited promotion", "other"] = Field(
        description="The type of spam."
    )


class NotSpamDetails(BaseModel):
    summary: str = Field(description="A brief summary of the content.")
    is_safe: bool = Field(description="Whether the content is safe for all audiences.")


class ModerationResult(BaseModel):
    decision: Union[SpamDetails, NotSpamDetails]


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """
Please moderate the following content and provide a decision.
Content: 'Congratulations! You''ve won a free cruise to the Bahamas. Click here to claim your prize: www.definitely-not-a-scam.com'
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": ModerationResult.model_json_schema(),
    },
)

result = ModerationResult.model_validate_json(response.text)
print(result)
