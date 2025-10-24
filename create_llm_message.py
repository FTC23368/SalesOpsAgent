from langchain_core.messages import SystemMessage, BaseMessage

def create_llm_msg(system_prompt: str, message_history: list[BaseMessage]):
    resp = []
    resp.append(SystemMessage(content=system_prompt))
    resp.extend(message_history)
    return resp