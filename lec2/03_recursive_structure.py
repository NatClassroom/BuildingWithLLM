from google import genai
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class Employee(BaseModel):
    """Represents an employee in an organization."""

    name: str
    employee_id: int
    reports: List["Employee"] = Field(
        default_factory=list,
        description="A list of employees reporting to this employee.",
    )


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """
Generate an organization chart for a small team.
The manager is Alice, who manages Bob and Charlie. Bob manages David.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": Employee.model_json_schema(),
    },
)

employee = Employee.model_validate_json(response.text)
print(employee)
