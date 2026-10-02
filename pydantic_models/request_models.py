from pydantic import BaseModel


class LLMInteractionModel(BaseModel):
    image_path: str
    src_age_estimate: str
    src_recognition_results: str