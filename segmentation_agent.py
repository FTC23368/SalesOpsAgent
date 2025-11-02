from langchain_core.messages import BaseMessage
from prompt_store import get_prompt
from create_llm_message import create_llm_msg

class SegmentationAgent:
    def __init__(self, model):
        self.model = model

    def generate_response(self, message_history: list[BaseMessage]):
        user_query = message_history[-1].content
        segmentation_prompt = get_prompt("segmentation").format(user_query=user_query)
        llm_messages = create_llm_msg(segmentation_prompt, message_history)
        return self.model.stream(llm_messages)

    def segmentation_agent(self, state: dict) -> dict:
        return {
            "lnode": "segmentation_agent",
            "incremental_response": self.generate_response(state["message_history"]),
            "category": "segmentation"
        }
