import os
from typing import Annotated

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sentiment: str


def preprocess(state: State) -> State:
    message = state["messages"][-1]
    message.content = message.content.strip()
    return state


def analyze_sentiment(state: State) -> State:
    text = state["messages"][-1].content.lower()
    positive_words = {"good", "great", "excellent", "happy", "love", "awesome"}
    negative_words = {"bad", "poor", "sad", "hate", "awful", "terrible"}

    if any(word in text for word in positive_words):
        sentiment = "positive"
    elif any(word in text for word in negative_words):
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment}


def build_graph(model_name: str) -> object:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except FileNotFoundError:
            api_key = None
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it as an environment variable "
            "or add it to .streamlit/secrets.toml."
        )

    llm = ChatGroq(api_key=api_key, model_name=model_name)

    def chatbot(state: State) -> dict[str, list[BaseMessage]]:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def logger(state: State) -> State:
        st.session_state.last_sentiment = state.get("sentiment", "neutral")
        return state

    builder = StateGraph(State)
    builder.add_node("preprocess", preprocess)
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("chatbot", chatbot)
    builder.add_node("logger", logger)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "chatbot")
    builder.add_edge("chatbot", "logger")
    builder.add_edge("logger", END)
    return builder.compile()


@st.cache_resource(show_spinner=False)
def get_graph(model_name: str) -> object:
    return build_graph(model_name)


st.set_page_config(page_title="LangGraph Sentiment Chat", page_icon="💬")
st.title("LangGraph Sentiment Chat")
st.caption("Preprocess → analyze sentiment → generate response → log")

model_name = st.sidebar.text_input(
    "Groq model",
    value=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sentiment" not in st.session_state:
    st.session_state.last_sentiment = "neutral"

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "user":
            st.caption(f"Sentiment: {message['sentiment'].title()}")

if prompt := st.chat_input("Send a message"):
    try:
        graph = get_graph(model_name)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = graph.invoke(
                    {"messages": [("user", prompt)], "sentiment": "neutral"}
                )
            response = result["messages"][-1].content
            st.markdown(response)

        sentiment = result.get("sentiment", "neutral")
        st.session_state.chat_history.extend(
            [
                {"role": "user", "content": prompt, "sentiment": sentiment},
                {"role": "assistant", "content": response},
            ]
        )
        st.rerun()
    except Exception as exc:
        st.error(str(exc))
