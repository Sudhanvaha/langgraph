import streamlit as st
from langgraph_backend import chatbot,llm
from langchain_core.messages import BaseMessage,HumanMessage
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




# ***********Session setup************************
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads']=[]
    
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()
    add_thread(st.session_state['thread_id'])

if 'thread_details' not in st.session_state:
    st.session_state['thread_details']={}

# *************************Sidebar UI*************
st.sidebar.title('Langgraph chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()




st.sidebar.header('My conversation')
for thread_id in reversed(st.session_state['chat_threads']):
    thread_name = st.session_state['thread_details'].get(
    thread_id,
    f"Chat {thread_id}"
    )


    if st.sidebar.button(str(thread_name)):
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
    
    
    title_response = llm.invoke(
    f"Based on this message: {user_input}, generate a short chat title of 2 or 3 words only"
    )

    title = str(title_response.content).strip()
                              
    st.session_state['thread_details'][st.session_state['thread_id']]=title
    #first add the message to the message history
    st.session_state['message_history'].append({'role':'user','content':user_input})

    with st.chat_message("user"):
        st.text(user_input)

     
    CONFIG={'configurable':{'thread_id':st.session_state['thread_id']}}
    # 
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {
                    'messages': [HumanMessage(content=user_input)]
                },
                config={
                    'configurable': {
                        'thread_id': str(st.session_state['thread_id'])
                    }
                },
                stream_mode='messages'
            )
        )
            

    st.session_state['message_history'].append({'role':'assistant','content':ai_message})



    