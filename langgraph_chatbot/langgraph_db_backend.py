from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3


from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model='llama-3.1-8b-instant')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)




graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
def retrive_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]['thread_id'])

    return list(all_threads)

conn.execute(
    """
        create table if not exists thread_details(
        thread_id str primary key,
        thread_title str)
    """
)

def add_thread_details(thread_id,title):
    conn.execute(
        """
        Insert into thread_details (thread_id,thread_title)
        values (?,?)
        """,(thread_id,title)
    )
    conn.commit()

def load_all_titles():
    cursor=conn.execute(
        """
            select thread_id,thread_title from thread_details
        """
    )
    rows=cursor.fetchall()
    thread_details=dict(rows)
    return thread_details