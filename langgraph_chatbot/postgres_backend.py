from __future__ import annotations

import os
import sqlite3
import psycopg
from psycopg.rows import dict_row
import tempfile
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated,Dict,Optional,Any
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langchain_groq import ChatGroq
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages,RemoveMessage
# import sqlite3
from langgraph.checkpoint.postgres import PostgresSaver


from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document


from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------
# LLM + Embeddings
#-----------------------

llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    
)

#------------------------------------------
# PDF retriever store(per thread)
#-----------------------------------
THREAD_RETRIEVERS:Dict[str,Any]={}
THREAD_METADATA:Dict[str,dict]={}

THREAD_YOUTUBE_RETRIEVERS:Dict[str,Any]={}
THREAD_YOUTUBE_METADATA:Dict[str,dict]={}


def get_retriever(thread_id:Optional[str]):
    """ Fetch the retriever for a thread if its available"""
    if thread_id and thread_id in THREAD_RETRIEVERS:
        return THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes:bytes,thread_id:str,filename:Optional[str]=None)->dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for Ingestion")
    
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path=temp_file.name

    try:
        loader=PyPDFLoader(temp_path)
        docs=loader.load()

        splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,chunk_overlap=200,separators=["\n\n", "\n", " ", ""]
        )
        chunks=splitter.split_documents(docs)
        if len(docs)==0 or len(chunks)==0:
            raise ValueError("No text could be extracted from the PDF.")

        vector_store=FAISS.from_documents(chunks,embeddings)
        retriever=vector_store.as_retriever(
            search_type="similarity",search_kwargs={"k":4}
        )

        THREAD_RETRIEVERS[str(thread_id)]=retriever
        THREAD_METADATA[str(thread_id)]={
            "filename":filename or os.path.basename(temp_path),
            "documents":len(docs),
            "chunks":len(chunks)

        }

        
        
        return{
            "filename":filename or os.path.basename(temp_path),
            "documents":len(docs),
            "chunks":len(chunks)
        }
    
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def get_youtube_retriever(thread_id:Optional[str]):
    """ Fetch the retriever for a thread if its available"""
    if thread_id and thread_id in THREAD_YOUTUBE_RETRIEVERS:
        return THREAD_YOUTUBE_RETRIEVERS[thread_id]
    return None

def ingest_youtube(url:str,thread_id:str)->dict:
    loader=YoutubeLoader.from_youtube_url(
    url,
    add_video_info=False,
    language=["en","hi","kn","ta"]
    )

    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(docs)
    

    vector_store=FAISS.from_documents(chunks,embeddings)
    retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
    THREAD_YOUTUBE_RETRIEVERS[str(thread_id)]=retriever
    THREAD_YOUTUBE_METADATA[str(thread_id)]={
        'url':url,
        "chunks":len(chunks),
        
    }

    return {
        'url':url,
        'chunks':len(chunks)
    }

#........................................
#tools
#.........................................

duckduckgo=DuckDuckGoSearchRun(region="us-en")
@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    Use only for recent/current information.
    """
    results= duckduckgo.run(query)
    return f"""
    Web search results:
    {results}

    Use ONLY this information to answer.
    """

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

@tool
def rag_tool(query:str,thread_id:Optional[str]=None)->dict:

    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """

    retriever=get_retriever(thread_id)
    if retriever is None:
        return{
            "error":"NO document indexed for this chat.Upload a PDF first",
            "query":query,
            }
    
    result=retriever.invoke(query)
    context=[doc.page_content for doc in result]
    metadata=[doc.metadata for doc in result]

    return {
        "query":query,
        "context":context,
        "metadata":metadata,
        "source_file":THREAD_METADATA.get(str(thread_id),{}).get('filename')
    }


@tool
def youtube_rag_tool(query:str,thread_id:str)->dict:
    """
    Retrieve relevant information from the provided url,and answer the question of the user accordingly
    """
    retriever=get_youtube_retriever(thread_id)
    if retriever is None:
        return{
            "error":"NO youtube video indexed for this chat.Upload a PDF first",
            "query":query,
            }


    result=retriever.invoke(query)
    context=[doc.page_content for doc in result]
    metadata=[doc.metadata for doc in result]

    return{
        "query":query,
        "context":context,
        "metadata":metadata,
        "source": THREAD_YOUTUBE_METADATA.get(str(thread_id), {}).get('url')
    }



#---------------------------------------------------
# Make tool list
tools = [get_stock_price, duckduckgo_search, calculator,rag_tool,youtube_rag_tool]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

#---------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # summary:str

# graph nodes
def chat_node(state: ChatState,config=None):
    """LLM node that may answer or request a tool call."""
    thread_id=None
    if config and isinstance(config,dict):
        thread_id=config.get("configurable",{}).get("thread_id")
    has_youtube = thread_has_youtube(thread_id)
    youtube_status = (
    "A YouTube video is already indexed for this thread."
    if has_youtube
    else
    "No YouTube video is indexed yet."
    )
    system_message=SystemMessage(
        content= (
            # "You are a helpful assistant. For questions about the uploaded PDF, call "
            # "the `rag_tool` and include the thread_id "
            # f"`{thread_id}`. You can also use the web search, stock price, and "
            # "calculator tools when helpful. If no document is available, ask the user "
            # "to upload a PDF."

            f"""
You are a helpful AI assistant with access to tools for PDF RAG, YouTube RAG, web search, stock lookup, and calculations.

Current thread information:

* thread_id = "{thread_id}"
* PDF indexed: {thread_has_document(thread_id)}
* YouTube video indexed: {thread_has_youtube(thread_id)}

Follow these rules carefully:

---

## GENERAL BEHAVIOR

1. Prefer answering directly whenever possible.

2. Do NOT use tools for:

   * greetings
   * casual conversation
   * simple explanations
   * coding help that does not require external data
   * general knowledge questions
   * follow-up conversational responses
   * basic reasoning tasks

3. Use tools ONLY when they are truly necessary.

---

## PDF RAG TOOL RULES

4. Use `rag_tool` ONLY when:

   * the user asks about the uploaded PDF
   * the answer depends on document content
   * the question references:

     * "the PDF"
     * "the document"
     * "this file"
     * "uploaded file"
     * "summarize the PDF"
     * "what does the document say"

5. When calling `rag_tool`, ALWAYS include:
   thread_id = "{thread_id}"

6. If no PDF is indexed and the user asks PDF-related questions:
   politely ask the user to upload a PDF first.

---

## YOUTUBE RAG TOOL RULES

7. Use `youtube_rag_tool` ONLY when:

   * the user asks about the indexed YouTube video
   * the answer depends on transcript/video content
   * the user refers to:

     * "this video"
     * "the video"
     * "YouTube video"
     * "summarize the video"
     * "what did he say"
     * "topics discussed"
     * "timestamps"
     * "explain the video"
     * "summarize it"

8. When calling `youtube_rag_tool`, ALWAYS include:
   thread_id = "{thread_id}"

9. If a YouTube video is already indexed:

   * NEVER ask the user for the URL again
   * directly use `youtube_rag_tool`

10. If no YouTube video is indexed and the user asks video-related questions:
    politely ask the user to share a YouTube URL first.

---

## WEB SEARCH RULES

11. Use `duckduckgo_search` ONLY when:

* the user asks for current/latest/recent information
* real-time information is required
* the user explicitly asks to search the web
* the answer cannot be reliably answered from existing knowledge

12. NEVER use web search for:

* simple coding questions
* explanations
* greetings
* math
* content already available in PDF/video context

---

## CALCULATOR RULES

13. Use `calculator` ONLY for actual mathematical calculations.

---

## STOCK TOOL RULES

14. Use `get_stock_price` ONLY for stock market related questions.

---

## TOOL RESPONSE RULES

15. After receiving tool results:

* answer ONLY from the tool output
* do NOT hallucinate
* do NOT speculate
* do NOT add warnings/disclaimers
* do NOT mention knowledge cutoff
* keep responses concise and accurate

16. If tool output is insufficient, say:
    "I could not find reliable information."

17. Never use web search if the answer already exists in:

* uploaded PDF context
* indexed YouTube transcript context

18. Always prioritize:
    PDF/YouTube context > direct knowledge > web search

"""

        )
    )
    
    
    messages = [system_message,*state['messages']]

    # if state["summary"]:
    #     messages.append(state["summary"])
    
    response = llm_with_tools.invoke(messages,config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)  # Executes tool calls


#---------------------------
# Checkpointer
#---------------------------
# conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)
DB_URI="postgresql://postgres:postgres@localhost:5432/postgres_chatbot"
conn = psycopg.connect(
    DB_URI,
    autocommit=True,
    row_factory=dict_row
)
checkpointer=PostgresSaver(conn=conn)
checkpointer.setup()


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

with conn.cursor() as cur:
    cur.execute(
        """
            create table if not exists thread_details(
            thread_id TEXT primary key,
            thread_title TEXT)
        """
    )

def add_thread_details(thread_id,title):
    with conn.cursor() as cur:
        cur.execute(
            """
            Insert into thread_details (thread_id,thread_title)
            values (%s,%s)
            """,(thread_id,title)
        )
        cur.commit()

def load_all_titles():
    with conn.cursor() as cur:
        cur.execute(
            """
                select thread_id,thread_title from thread_details
            """
        )
        rows=cur.fetchall()
    thread_details=dict(rows)
    return thread_details

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return THREAD_METADATA.get(str(thread_id), {})

def thread_has_youtube(thread_id: str) -> bool:
    return str(thread_id) in THREAD_YOUTUBE_RETRIEVERS

def thread_youtube_metadata(thread_id: str) -> dict:
    return THREAD_YOUTUBE_METADATA.get(str(thread_id), {})

# if __name__ == "__main__":

#     thread_id = "demo-thread"

#     while True:
#         user_input = input("You: ")

#         if user_input.lower() in ["exit", "quit"]:
#             break

#         response = chatbot.invoke(
#             {
#                 "messages": [
#                     HumanMessage(content=user_input)
#                 ]
#             },
#             config={
#                 "configurable": {
#                     "thread_id": thread_id
#                 }
#             }
#         )

#         print("\nAI:", response["messages"][-1].content)