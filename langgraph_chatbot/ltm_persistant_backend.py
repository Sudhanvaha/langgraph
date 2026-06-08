from __future__ import annotations

import traceback
import os
import sqlite3
import psycopg
from psycopg.rows import dict_row
import tempfile
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated,Dict,Optional,Any,Literal,List
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
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
import faiss
import numpy as np
import pickle
import json
from datetime import datetime, timezone
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document


#ltm
from langgraph.store.postgres import PostgresStore
import psycopg
from langgraph.store.base import BaseStore
from pydantic import BaseModel,Field
from langchain_core.runnables import RunnableConfig


from dotenv import load_dotenv

load_dotenv()
base_path=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\pdf"
base_path_yt=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\youtube"

# ------------------------------------------
# LLM + Embeddings
#-----------------------
memory_llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0)

llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    
)

#------------------------------------------
# PDF retriever store(per thread)
#-----------------------------------
THREAD_RETRIEVERS:Dict[str,Any]={}
THREAD_METADATA:Dict[str,dict]={}

THREAD_YOUTUBE_VECTORSTORE:Dict[str,Any]={}
THREAD_YOUTUBE_METADATA:Dict[str,dict]={}


#no use of this since we are fetching it from storage/pdf/index.faiss,index.json,index.pkl
def get_retriever(thread_id:Optional[str]):
    """ Fetch the retriever for a thread if its available"""
    if thread_id and thread_id in THREAD_RETRIEVERS:
        return THREAD_RETRIEVERS[thread_id]
    
    file_path = os.path.join(base_path, thread_id)
    if not os.path.exists(file_path):
        return None

    #Disk load
    pdf_index=faiss.read_index(
        os.path.join(file_path,"index.faiss")
    )

    with open(os.path.join(file_path,"index.pkl"),"rb") as f:
        pdf_chunks=pickle.load(f)
    with open(os.path.join(file_path,"index.json"),"r") as f:
        metadata=json.load(f)

    data = {
        "index": pdf_index,
        "chunks": pdf_chunks,
        "metadata": metadata
    }

    # Store in cache
    THREAD_RETRIEVERS[thread_id] = data

    return data

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
        
        texts=[doc.page_content for doc in chunks]
        document_embeddings=embeddings.embed_documents(texts)
        
        pdf_index = faiss.IndexFlatL2(len(document_embeddings[0]))
        pdf_index_embeddings= pdf_index.add(np.array(document_embeddings,dtype=np.float32))

        file_path=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\pdf\{thread_id}"

        os.makedirs(file_path, exist_ok=True)
        
        faiss.write_index(pdf_index, os.path.join(file_path, "index.faiss"))

        
        with open( os.path.join(file_path, "index.pkl"),"wb") as f:
            pickle.dump(chunks,f)


        metadata={
            "thread_id":thread_id,
            "filename":filename,
            "documents": len(docs),
            "num_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        THREAD_METADATA[str(thread_id)] = metadata
        with open(os.path.join(file_path,"index.json"),"w") as f:
            json.dump(metadata, f, indent=2)

        THREAD_RETRIEVERS[thread_id] = {
            "index": pdf_index,
            "chunks": chunks,
            "metadata": metadata
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
    if thread_id and thread_id in THREAD_YOUTUBE_VECTORSTORE:
        return THREAD_YOUTUBE_VECTORSTORE[thread_id].as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    file_path=os.path.join(base_path_yt,thread_id)
    if not file_path:
        return None
    
    vector_store = FAISS.load_local(
    file_path,
    embeddings,
    allow_dangerous_deserialization=True
    )

    retriever=vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":4,},
        
    )
    THREAD_YOUTUBE_VECTORSTORE[str(thread_id)] = vector_store
    return retriever

def ingest_youtube(url:str,thread_id:str)->dict:
    loader=YoutubeLoader.from_youtube_url(
    url,
    add_video_info=False,
    language=["en","hi","kn","ta"]
    )

    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_documents(docs)
    file_path=os.path.join(base_path_yt,thread_id)

    vector_store=FAISS.from_documents(chunks,embeddings)
    vector_store.save_local(os.path.join(file_path))

    # retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
    THREAD_YOUTUBE_VECTORSTORE[str(thread_id)]=vector_store
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

    if not retriever:
        return{
            "error":"NO document indexed for this chat.Upload a PDF first",
            "query":query,
            }
    
    pdf_index = retriever["index"]
    pdf_chunks = retriever["chunks"]
    pdf_metadata = retriever["metadata"]

    query_embedding=embeddings.embed_query(query)
    query_vector=np.array(
        [query_embedding],
        dtype=np.float32
    )

    distances, indices = pdf_index.search(query_vector, k=4)

    context = []

    for dist, idx in zip(distances[0], indices[0]):
        if dist < 1.5:      # tune threshold
            context.append(pdf_chunks[idx].page_content)



    return {
        "query":query,
        "context":"\n\n".join(context),
        "metadata":pdf_metadata,
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
    summary:str=""

class MemoryItem(BaseModel):
    category: str = Field(description="Memory category")
    key: str = Field(description="Unique attribute key")
    value: str = Field(description="Actual memory value")
    text: str = Field(description="Human readable memory")
    action: Literal["create", "update", "duplicate"]

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


memory_extractor =memory_llm.with_structured_output(MemoryDecision)
MEMORY_PROMPT = """
You are a user-memory extraction system.

Existing user memories are provided in:

{user_details_content}

Format of user_details_content:

* user_details_content is a list[dict].
* Each dictionary represents an existing memory record.
* Compare newly extracted information against these records before deciding whether it is new, updated, or already known.

Example:

[
{{
"category": "identity",
"key": "name",
"value": "Sudhanva",
"text": "User's name is Sudhanva."
}},
{{
"category": "preference",
"key": "language",
"value": "Python",
"text": "User prefers Python."
}}
]

Your task:
Analyze the user's latest message and determine whether any information should be stored as long-term memory.

Store only information that is likely to remain useful across future conversations, such as:

* Identity information
* Stable preferences
* Long-term goals
* Ongoing projects
* Recurring life context
* Persistent responsibilities
* Frequently referenced personal context

Do NOT store:

* Temporary emotions, moods, or feelings
* One-time requests
* Short-lived plans
* Session-specific context
* Information not explicitly stated by the user
* Assumptions, guesses, or inferred facts
* Generic conversational details with little future value

Memory writing rules:

* Every memory must contain exactly one fact.
* Keep memories concise and atomic.
* Write all memories as standalone third-person facts.
* Normalize first-person statements into canonical form.
* Do not include conversational wording.

Examples:

User: "I am Sudhanva"
Memory text: "User's name is Sudhanva."

User: "I live in Bangalore"
Memory text: "User lives in Bangalore."

User: "I like Python"
Memory text: "User prefers Python."

User: "I am building a RAG chatbot"
Memory text: "User is building a RAG chatbot."

MemoryItem field definitions:

* category:
  High-level memory category such as:
  identity, preference, goal, project, relationship, location, work, education, context

* key:
  Stable attribute identifier.
  Examples:
  name
  city
  favorite_language
  current_project

* value:
  Canonical value associated with the key.

* text:
  Human-readable memory written as a standalone third-person fact.

* action:
  Must be one of:

  * create
  * update
  * duplicate

Action selection rules:

create

* The fact does not exist in user_details_content.
* The information is memory-worthy and should be stored.

update

* An existing memory with the same key or meaning should be replaced or corrected.
* Use when the user provides newer, more accurate, or changed information.

duplicate

* The same fact already exists in user_details_content.
* The new information does not add anything materially different.

Deduplication rules:

* Compare semantic meaning, not exact wording.
* If an equivalent memory already exists, use duplicate.
* If a memory conflicts with an existing memory, use update.
* If the fact is entirely new, use create.

Decision rules:

* should_write = true if at least one memory has action=create or action=update.
* should_write = false if:

  * no memory-worthy information exists, or
  * all extracted memories are duplicates.

Output rules:

* Output must strictly follow the MemoryDecision schema.
* If should_write=false, return an empty memories list.
* Do not include explanations, reasoning, or additional fields.
* Return only schema-compliant data.

"""

# ----------------------------
# 2) System prompt
# ----------------------------
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Nitish...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user’s memory (which may be empty) is provided as: {user_details_content}
"""



# ----------------------------
#  Node : remember
# ----------------------------
def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["thread_id"]
    ns = ("user", user_id, "details")

    # existing memory
    existing = store.search(ns)
    existing_items=[]
    for item in existing:
        existing_items.append(
            
                {
                "category":item.value['category'],
              "key":item.value['key'],
              "value":item.value["value"],
              "text":item.value["text"]
              }
              
              )
    # print(existing_items)
    decision:MemoryDecision=memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing_items)),
            {'role':'user','content':state["messages"][-1].content}
        ]
    )


    if decision.should_write:
        for mem in decision.memories:
            if mem.action=="create" or mem.action=="update":
                
                store.put(
                    ns,
                    str(mem.category+mem.key),
                    {
                        "category":mem.category,
                        "key":mem.key,
                        "value":mem.value,
                        "text":mem.text
                    }
                )

    return {}  # no message change



def summarize_conversation(state:ChatState):
    existing_summary=state["summary"]
    if existing_summary:
        prompt=(
             f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
         prompt=("summarize the conversation above")
    
    message_for_summary=state["messages"]+[HumanMessage(content=prompt)]
    response=llm.invoke(message_for_summary)

    messages_to_delete=state['messages'][:-4]
    return {
        "summary":response.content,
        "messages":[RemoveMessage(id=m.id) for m in messages_to_delete]
    }

def should_summarize(state:ChatState)->bool:
    return len(state["messages"])>6


def route_after_chat(state:ChatState):

    has_tools = tools_condition(state) == "tools"
    needs_summary = should_summarize(state)
    
    if has_tools and needs_summary:
        return "summarize_then_tools"
    
    if has_tools:
        return "tools"
    
    if needs_summary:
        return "summarize"
    return END

# graph nodes
def chat_node(state: ChatState,config:RunnableConfig,store:BaseStore):
    """LLM node that may answer or request a tool call."""
    thread_id=config["configurable"]["thread_id"]
    ns=("user",thread_id,"details")

    items = store.search(ns)
    
    user_details=[]
    for item in items:
        user_details.append(
                {
                "category":item.value['category'],
              "key":item.value['key'],
              "value":item.value["value"],
              "text":item.value["text"]
              }
              
              )
        
    system_msg_ltm = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(
            user_details_content=user_details or "(empty)"
        )
    )
    


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
    messages=[system_msg_ltm]
    if state["summary"]:
        messages.append(
            SystemMessage(
                content=f"Conversation summary:\n{state['summary']}"
                )
            
        )
    messages.append(system_message)
    messages.extend(state['messages'])
    
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        
        traceback.print_exc()
        response = AIMessage(
        content=f"An error occurred: {str(e)}"
    )
        
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


DB_URI_LTM="postgresql://postgres:postgres@localhost:5442/postgres_ltm_store"
ltm_conn=psycopg.connect(
    DB_URI_LTM,
    autocommit=True
)
ltm_store=PostgresStore(conn=ltm_conn)
ltm_store.setup()


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools",tool_node)
graph.add_node("summarize",summarize_conversation)
graph.add_node("summarize_then_tools", summarize_conversation)
graph.add_node("remember",remember_node)


graph.add_edge(START,"remember")
graph.add_edge("remember", "chat_node")
graph.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {
        "summarize_then_tools":"summarize_then_tools",
        "tools":"tools",
        "summarize":"summarize",
        END: END
    }
)

graph.add_edge("summarize",END)
graph.add_conditional_edges(
    "summarize_then_tools",
    lambda _: "tools",
    {"tools": "tools"}
)


graph.add_edge("tools","chat_node")


chatbot = graph.compile(checkpointer=checkpointer,store=ltm_store)




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
            INSERT INTO thread_details (thread_id, thread_title)
            VALUES (%s, %s)
            ON CONFLICT (thread_id)
            DO UPDATE SET
            thread_title = EXCLUDED.thread_title
            """,(thread_id,title)
        )
        

def load_all_titles():
    with conn.cursor() as cur:
        cur.execute(
            """
                select thread_id,thread_title from thread_details
            """
        )
        rows = cur.fetchall()
    
    thread_details = {row["thread_id"]: row["thread_title"] for row in rows}
    return thread_details

def thread_has_document(thread_id: str) -> bool:
    BASE_PATH=fr"C:\Users\sudha\Desktop\My_workspace\langgraph\langgraph_chatbot\Multi-Source_AI_Assistant\storage\pdf"
    path = os.path.join(BASE_PATH, thread_id, "index.faiss")
    return os.path.exists(path)


def thread_document_metadata(thread_id: str) -> dict:
    return THREAD_METADATA.get(str(thread_id), {})

def thread_has_youtube(thread_id: str) -> bool:
    return str(thread_id) in THREAD_YOUTUBE_VECTORSTORE

def thread_youtube_metadata(thread_id: str) -> dict:
    return THREAD_YOUTUBE_METADATA.get(str(thread_id), {})
