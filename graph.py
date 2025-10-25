import streamlit as st
from pydantic import BaseModel
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from prompt_store import get_prompt
from create_llm_message import create_llm_msg
from smalltalk_agent import SmallTalkAgent
from clarify_agent import ClarifyAgent
from policy_agent import PolicyAgent
from quota_agent import QuotaAgent
from segmentation_agent import SegmentationAgent
from saleshierarchy_agent import SalesHieararchyAgent

class AgentState(TypedDict):
    lnode: str
    category: str
    incremental_response: str
    initial_response: str
    message_history: list[BaseMessage]

class Category(BaseModel):
    category: str

VALID_CATEGORIES = ["smalltalk", "clarify", "policy", "quota", "segmentation", "saleshierarchy"]

class SalesOpsAgent:
    def __init__(self, api_key):
        self.model = ChatOpenAI(model=st.secreats['OPENAI_MODEL'], api_key=api_key)

        self.smalltalk_agent_class = SmallTalkAgent(self.model)
        self.clarify_agent_class = ClarifyAgent(self.model)
        self.policy_agent_class = PolicyAgent(self.model)
        self.quota_agent_class = QuotaAgent(self.model)
        self.segmentation_agent_class = SegmentationAgent(self.model)
        self.saleshierarchy_agent_class = SalesHierarchyAgent(self.model)

        workflow = StateGraph(AgentState)

        workflow.add_node("classifier", self.initial_classifier)
        workflow.add_node("smalltalk", self.smalltalk_agent_class.smalltalk_agent)
        workflow.add_node("clarify", self.clarify_agent_class.clarify_agent)
        workflow.add_node("policy", self.policy_agent_class.policy_agent)
        workflow.add_node("quota", self.quota_agent_class.quota_agent)
        workflow.add_node("segmentation", self.segmentation_agent_class.segmentation_agent)
        workflow.add_node("saleshierarchy", self.saleshierarchy_agent_class.saleshierarchy_agent)

        workflow.add_conditional_edges("classifier", self.main_router)
        workflow.add_edge(START, "classifier")
        workflow.add_edge("smalltalk", END)
        workflow.add_edge("clarify", END)
        workflow.add_edge("quota", END)
        workflow.add_edge("segmentation", END)
        workflow.add_edge("saleshierarchy", END)

        self.graph = workflow.compile()

    def initial_classifier(self, state: AgentState):
        print(f"initial classifier")
        classifier_prompt = get_prompt("classifier")
        llm_messages = create_llm_msg(classifier_prompt, state["message_history"])
        llm_response = self.model.with_structured_output(Category).invoke(llm_messages)
        category = llm_response.category
        return {
            "lnode": "initial classifier",
            "category": category,
        }

    def main_router(self, state: AgentState):
        main_category = state["category"]
        if main_category in VALID_CATEGORIES:
            return main_category
        else:
            print(f"Missing category: {main_category}")
            return END