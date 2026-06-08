from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model='llama-3.1-8b-instant')

#tools.................................
search_tool=DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num:float,second_num:float,operation:str)->dict:
    """
    Perform a basic arithmetic operation on two numbers.
    and make sure to convert both the num to float
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


#---------------------------------------------------
# Make tool list
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

#---------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# graph nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)  # Executes tool calls

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)




graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

#helper functions --------------------------------------
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


