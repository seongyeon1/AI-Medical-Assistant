import os
import streamlit as st
import requests
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd

from secrets import serviceKey
# Set your API key and endpoint
# serviceKey='Decoding key'
LANGSERVE_ENDPOINT = "http://localhost:8000/llm/c/N4XyA"

# Embedding 설정
USE_BGE_EMBEDDING = True

# if not USE_BGE_EMBEDDING:
#     os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 프롬프트 설정
RAG_PROMPT_TEMPLATE = """당신은 친절한 AI입니다. 증상을 기반으로 질병을 예측하고, 해당 질병의 진료과를 추천합니다. 문맥을 사용하여 질문에 답하세요.

요구사항:
- 예상 질병에 대한 설명과 대처방안에 대해서 얘기해 줍니다. (5개 정도까지 가능성이 높은 것을 얘기해줍니다)
- 성별, 나이에 따른 다른 결과값
- 한글로 작성

Question: {question} 
Context: {context} 
Answer:"""

# Read symptoms data
def load_symptoms_data(file_path):
    return pd.read_csv(file_path, delimiter='\t')

symptoms_df = load_symptoms_data('symptoms.txt')

# Streamlit setup
st.set_page_config(page_title="질병 예측 및 병원 추천", page_icon="🏥")
st.title("질병 예측 및 병원 추천 시스템")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

def print_history():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{os.path.basename(file_path)}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    if USE_BGE_EMBEDDING:
        model_name = 'jhgan/ko-sroberta-multitask'
        model_kwargs = {"device": "mps"}
        encode_kwargs = {"normalize_embeddings": True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

retriever = embed_file('symptoms.txt')
llm = RemoteRunnable(LANGSERVE_ENDPOINT)

print_history()

if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        if retriever is not None:
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            answer = rag_chain.stream(user_input)
            chunks = []
            for chunk in rag_chain.stream(user_input):
                chunks.append(chunk)
            answer = "".join(chunks)
            st.markdown(answer)
            add_history("assistant", "".join(chunks))
        else:
            st.write("문서 임베딩에 문제가 발생했습니다.")
