from langchain_core.messages import BaseMessage
from prompt_store import get_prompt
from create_llm_message import create_llm_msg

class SalesHierarchyAgent:
    def __init__(self, model):
        self.model = model

    def generate_response(self, message_history: list[BaseMessage]):
        user_query = message_history[-1].content
        saleshierarchy_prompt = get_prompt("saleshierarchy").format(user_query=user_query)
        llm_messages = create_llm_msg(saleshierarchy_prompt, message_history)
        return self.model.stream(llm_messages)

    def saleshierarchy_agent(self, state: dict) -> dict:
        return {
            "lnode": "saleshierarchy_agent",
            "incremental_response": self.generate_response(state["message_history"]),
            "category": "saleshierarchy"
        }
