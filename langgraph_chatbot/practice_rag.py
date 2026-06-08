from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='llama-3.1-8b-instant')

url="https://youtu.be/qYNweeDHiyU?si=qYDqlMWm6C_-8pMn"

loader=YoutubeLoader.from_youtube_url(
    url,
    add_video_info=False,
    language="en"
)

docs=loader.load()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=text_splitter.split_documents(docs)

embeddings=HuggingFaceEmbeddings()

vector_store=FAISS.from_documents(chunks,embeddings)


user_input=input("ask your query related to the youtube video")
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})

context=retriever.invoke(user_input)


response=llm.invoke(f"here's the user query{user_input} and here's the context {context} answer only if its present in the context else reply it was not present in video or i dont know")
print(response.content)
