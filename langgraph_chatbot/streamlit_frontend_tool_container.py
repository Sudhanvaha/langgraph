import streamlit as st
from langgraph_tool_backend import chatbot,llm,retrive_all_threads,add_thread_details,load_all_titles
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolMessage
import uuid



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

def current_thread_has_no_title(thread_id)->bool:
    if thread_id not in st.session_state['thread_details']:
        return True
    else:
        return False

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



# *************************Sidebar UI*************
st.sidebar.title('Langgraph chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()




st.sidebar.header('My conversation')
for thread_id in reversed(st.session_state['chat_threads']):
    title=st.session_state['thread_details'].get(
        thread_id,
        f"Chat{thread_id}"
    )
    if st.sidebar.button(title):
        st.session_state['thread_id']=thread_id
        messages=load_conversation(thread_id)

        temp_messages=[]
        
        for message in messages:
            if isinstance(message,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role':role,'content':message.content})

        st.session_state['message_history']=temp_messages

# *************************************************************


user_input=st.chat_input("Type here")

#loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])



if user_input:
    
    if current_thread_has_no_title(st.session_state['thread_id']):
        genereate_title(user_input)

    #first add the message to the message history
    st.session_state['message_history'].append({'role':'user','content':user_input})

    with st.chat_message("user"):
        st.text(user_input)

     
    # CONFIG={'configurable':{'thread_id':st.session_state['thread_id']}}
    CONFIG={'configurable':{'thread_id':st.session_state['thread_id']},
            "metadata":{
                "thread_id":st.session_state['thread_id']
            },
            "run_name":"chat_turn"
            }
    # 
    
    with st.chat_message('assistant'):

        with st.status("Thinking...", expanded=True) as status:

            def ai_only_stream():

                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [HumanMessage(content=user_input)]
                    },
                    config=CONFIG,
                    stream_mode="messages"
                ):

                    # show tool activity
                    if hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:

                        for tool_call in message_chunk.tool_calls:

                            tool_name = tool_call["name"]

                            status.write(f"🔧 Using tool: {tool_name}")

                    # stream only AI text
                    if isinstance(message_chunk, AIMessage):

                        if message_chunk.content:
                            yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

        status.update(
            label="Completed",
            state="complete",
            expanded=False
        )

    st.session_state['message_history'].append({'role':'assistant','content':ai_message})