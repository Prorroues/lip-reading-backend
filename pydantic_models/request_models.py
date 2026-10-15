from pydantic import BaseModel, Field


class LLMInteractionModel(BaseModel):
    image_path: str
    src_age_estimate: str
    src_recognition_results: str


class AssistantChatModel(BaseModel):
    text: str = Field(..., min_length=1, description="家居助手要处理的句子")