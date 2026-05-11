from pydantic import BaseModel, Field
from typing import List

class FinanceAgentResponse(BaseModel):
    question: str = Field(description="The user's original question.")
    answer: str = Field(description="The final answer to the question.")
    sources: List[str] = Field(description="List of document names or data sources used to answer the question.")
    tools_used: List[str] = Field(description="List of tools used by the agent.")
    confidence: str = Field(description="Confidence level of the answer: high, medium, or low.")
    timestamp: str = Field(description="ISO 8601 formatted timestamp of when the response was generated.")
