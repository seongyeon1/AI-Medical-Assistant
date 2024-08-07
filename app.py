import os
import streamlit as st
import requests
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
import pandas as pd

from utils.get_department import extract_first_department
# from utils.get_location import location_df

import json
import folium
from streamlit_folium import st_folium

# Set your API key and endpoint
LANGSERVE_ENDPOINT = "http://localhost:8000/llm/c/N4XyA"

# Embedding 설정
USE_BGE_EMBEDDING = True

if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 프롬프트 설정
RAG_PROMPT_TEMPLATE = """
당신은 친절한 AI 의사입니다. 증상을 기반으로 질병을 예측하고, 해당 질병의 진료과를 추천합니다. 
문맥을 사용하여 질문에 답하세요.

Question: {question}

요구사항:
- 예상 질병은 최대 3개까지 제시
- "진료과: 내과" 형식으로 제시
- 중요한 단어 강조
- 한글로 작성

Format (예시):
~~증상을 가지고 계시는 군요. 해당 증상에 따라 가능성이 높은 질병을 세 가지 알려드리겠습니다!
\n
1. 질병1
    - 증상 (10개 이하)
    - 원인
    - 대처방안
    - 진료과
2. 질병2 
    - 증상
    - 원인
    - 대처방안
    - 진료과 
3. 질병3
    - 증상
    - 원인
    - 대처방안
    - 진료과 
\n
추가적인 검진이 필요하시다면 **가능성이 가장 높은 진료과**에 방문하시어 정확한 검진 받으시길 바랍니다.
 
Context: {context} 
Answer:"""

# Streamlit setup
st.set_page_config(page_title="질병 예측 및 병원 추천", page_icon="🏥", layout="wide")

def main():
    st.title("질병 예측 및 병원 추천 시스템")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            ChatMessage(role="assistant", content="어떤 증상이 있으신가요?")
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
        loader = UnstructuredLoader(file_path)
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
                
                chunks = []
                try:
                    for chunk in rag_chain.stream(user_input):
                        if not isinstance(chunk, str):
                            raise TypeError(f"Expected chunk to be str but got {type(chunk).__name__}")
                        chunks.append(chunk)
                except Exception as e:
                    st.error(f"Error in processing the chain: {str(e)}")
                    chunks.append("")

                answer = "".join(chunks)
                formatted_answer = answer.replace("\n", "  \n")
                st.markdown(formatted_answer)
                add_history("assistant", answer)
                
                # 함수 호출 및 결과 출력
                first_department = extract_first_department(answer)
            
                with open('./data/department.json', 'r') as f:
                    data = json.load(f)

                st.session_state.first_department = first_department
                st.session_state.department = data.get(first_department, '')
            else:
                st.write("문서 임베딩에 문제가 발생했습니다.")


def create_map(df):
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    map_ = folium.Map(location=map_center, zoom_start=16)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['기관명']} ({row['종별코드명']})",
            tooltip=row['기관명']
        ).add_to(map_)

    return map_

def map_page():
    st.title("근처 병원 지도")
    st.write("아래 지도에서 병원 위치를 확인할 수 있습니다.")

    if "emdongNm" not in st.session_state:
        st.session_state.emdongNm = ""
    if "department" not in st.session_state:
        st.session_state.department = ""

    emdongNm = st.text_input("위치(읍면동)를 입력하세요 (예: 청담동):", st.session_state.emdongNm)
    department = st.text_input("진료과를 입력하세요 (예: 내과):", st.session_state.department)

    if st.button("병원 찾기"):
        if emdongNm and department:
            st.session_state.emdongNm = emdongNm
            st.session_state.department = department
            st.write(f"입력된 위치: {emdongNm}, 진료과: {department}")
            
            try:
                # CSV 파일 로드
                df = pd.read_csv('./data/hospital_db.csv')

                # 필터링
                if emdongNm:
                    c1 = df['읍면동'].str.contains(emdongNm)
                else:
                    c1 = True

                if department:
                    c2 = df['진료과'].str.contains(department)
                else:
                    c2 = True

                df_filtered = df[c1 & c2]

                # 필터링된 데이터 출력 (디버깅 목적)
                # st.write("필터링된 데이터프레임:", df_filtered)

                # 지도 생성 및 마커 추가
                if not df_filtered.empty:
                    map_ = create_map(df_filtered)
                #     # 지도 출력
                #     st_data = st_folium(map_, width=700, height=500)

                # else:
                #     st.write("해당 위치에 병원이 없습니다.")
            except Exception as e:
                st.write(f"데이터 처리 중 오류가 발생했습니다: {e}")

            st.caption("Hospital data on a map using Streamlit and Folium")
            st_data = st_folium(map_, width=700, height=500)    
        else:
            st.write("위치와 진료과를 모두 입력하세요.")
    else:
        if st.session_state.emdongNm and st.session_state.department:
            try:
                # CSV 파일 로드
                df = pd.read_csv('./data/hospital_db.csv')

                # 필터링
                if st.session_state.emdongNm:
                    c1 = df['읍면동'].str.contains(st.session_state.emdongNm)
                else:
                    c1 = True

                if st.session_state.department:
                    c2 = df['진료과'].str.contains(st.session_state.department)
                else:
                    c2 = True

                df_filtered = df[c1 & c2]

                # 필터링된 데이터 출력 (디버깅 목적)
                # st.write("필터링된 데이터프레임:", df_filtered)

                # 지도 생성 및 마커 추가
                if not df_filtered.empty:
                    map_ = create_map(df_filtered)
                    # 지도 출력
                    st_data = st_folium(map_, width=700, height=500)

                else:
                    st.write("해당 위치에 병원이 없습니다.")
            except Exception as e:
                st.write(f"데이터 처리 중 오류가 발생했습니다: {e}")

            st.caption("Hospital data on a map using Streamlit and Folium")
            st_data = st_folium(map_, width=700, height=500)

# 페이지 선택
page = st.sidebar.selectbox("페이지 선택", ["질병 예측", "병원 지도"])

if page == "질병 예측":
    main()
elif page == "병원 지도":
    map_page()
