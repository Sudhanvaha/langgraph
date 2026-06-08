import streamlit as st
from ltm_persistant_backend import (
    chatbot,
    llm,
    retrive_all_threads,
    add_thread_details,
    load_all_titles,
    thread_document_metadata,
    thread_has_document,
    ingest_pdf,
    ingest_youtube
    )
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolMessage
import uuid
import re


# *********************utility functions***********8




def generate_thread_id():
    thread_id=uuid.uuid4()
    
    return str(thread_id)

def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(thread_id)
    st.session_state['message_history']=[]

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state=chatbot.get_state(
        config={
            'configurable':{
                'thread_id':thread_id
                }
                }
            )
    if not state.values:
        return []
    
    return state.values["messages"]


def genereate_title(user_input):

    thread_title=llm.invoke(f"here is the user input :-{user_input} based on this provide me a title with 2 or 3 words not more than that,the reply should only consists of 2 or 3 words not more than that")
    title=str(thread_title.content).strip()
    st.session_state['thread_details'][st.session_state['thread_id']]=title

    add_thread_details(st.session_state['thread_id'],title)
    st.session_state['thread_details'] = load_all_titles()

def current_thread_has_no_title(thread_id)->bool:
    if thread_id not in st.session_state['thread_details']:
        return True
    else:
        return False
    

def extract_youtube_url(text: str):

    pattern = r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[^\s]+)"

    match = re.search(pattern, text)

    if match:
        return match.group(0)

    return None


# ***********Session setup************************
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=retrive_all_threads()
    
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()
    add_thread(st.session_state['thread_id'])

if 'thread_details' not in st.session_state:
    st.session_state['thread_details']=load_all_titles()

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs']={}

thread_key=st.session_state['thread_id']
thread_docs=st.session_state['ingested_docs'].setdefault(thread_key,{})
threads=st.session_state["chat_threads"][::-1]
selected_thread=None



# *************************Sidebar UI*************


st.sidebar.title('Langgraph chatbot')
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button('New Chat',use_container_width=True):
    reset_chat()

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF..",expanded=True) as status_box:
            try:
                summary=ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                

                thread_docs[uploaded_pdf.name]=summary
                status_box.update(label="✅PDF indexed",state="complete",expanded=False)
            except ValueError as e:
                st.sidebar.error(str(e))
if thread_docs:
    latest_doc=list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}`"
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")


st.sidebar.subheader('My conversation')
for thread_id in reversed(st.session_state['chat_threads']):
    title = st.session_state['thread_details'].get(thread_id)
    # Only show threads that have a real title — skip brand-new untitled threads
    if not title:
        continue
    if st.sidebar.button(title, key=f"thread_{thread_id}", use_container_width=True):
        st.session_state['thread_id']=thread_id
        messages=load_conversation(thread_id)

        temp_messages=[]
        
        for message in messages:
            if isinstance(message,HumanMessage):
                role='user'
            elif isinstance(message, AIMessage) and message.content:
                role='assistant'
            else:
                # Skip ToolMessages and empty AI messages (tool-call stubs)
                continue
            temp_messages.append({'role':role,'content':message.content})

        st.session_state['message_history']=temp_messages

# *************************************************************
st.title("Multi Utility Chatbot")

user_input=st.chat_input("Type here")

#loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])




if user_input:
    youtube_url=extract_youtube_url(user_input)
    if youtube_url:
        
        with st.chat_message("assistant"):
            with st.status("processing youtube video..",expanded=True):
                try:
                    summary=ingest_youtube(
                        youtube_url,
                        thread_key
                    )
                    if summary:
                        st.success("YouTube video indexed successfully!")

                    st.session_state['message_history'].append({
                        'role': 'assistant',
                        'content': 'YouTube video indexed successfully!'
                    })
                except Exception as e:
                    st.error(str(e))
         # Remove URL from user message
        user_input = user_input.replace(youtube_url, "").strip()
    if not user_input:
        st.stop()


    if current_thread_has_no_title(st.session_state['thread_id']) or st.session_state['thread_details'].get(thread_key)==thread_key:
        genereate_title(user_input)

    #first add the message to the message history
    st.session_state['message_history'].append({'role':'user','content':user_input})

    with st.chat_message("user"):
        st.text(user_input)

     
    CONFIG={
        'configurable':{'thread_id':st.session_state['thread_id']},
        "metadata":{
            "thread_id":st.session_state['thread_id']
        },
        "run_name":"chat_turn"
    }
    
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        with st.status("Thinking...", expanded=True) as status:

            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "summary":""
                    },
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    # Show tool activity inside the status box
                    if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                        for tool_call in message_chunk.tool_calls:
                            tool_name = tool_call["name"]
                            status.write(f"🔧 Using tool: {tool_name}")

                    
                    langgraph_node = metadata.get("langgraph_node", "")
                    if langgraph_node != "chat_node":
                        continue
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content

            with message_placeholder:
                ai_message = st.write_stream(ai_only_stream)

            status.update(
                label="Completed",
                state="complete",
                expanded=False
            )

    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
