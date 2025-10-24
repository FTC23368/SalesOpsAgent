from langchain_core.messages import BaseMessage
from prompt_store import get_prompt
from create_llm_message import create_llm_msg

class ClarifyAgent:
    def __init__(self, model):
        self.model = model

    def generate_response(self, message_history: list[BaseMessage]):
        user_query = message_history[-1].content
        clarify_prompt = get_prompt("clarify").format(user_query=user_query)
        llm_messages = create_llm_msg(clarify_prompt, message_history)
        return self.model.stream(llm_messages)

    def clarify_agent(self, state: dict) -> dict:
        return {
            "lnode": "clarify_agent",
            "incremental_message": self.generate_response(state["message_history"]),
            "category": "clarify"
        }