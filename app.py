# 라이브러리 불러오기
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 환경 변수 로드
load_dotenv("../data/.env")
api_key = os.getenv("OPENAI_API_KEY")

# PDF 처리 함수
@st.cache_resource
def process_pdf():
    loader = PyPDFLoader("../data/2024_KB_부동산_보고서_최종.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 벡터스토어 초기화 (FAISS로 변환)
@st.cache_resource
def initialize_vectorstore():
    faiss_directory = "../faiss_db"
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)

    # 기존 faiss_db가 있으면 그대로 유지하면서 로드
    if os.path.exists(faiss_directory):
        faiss_db = FAISS.load_local(
            faiss_directory,
            embedding_function,
            allow_dangerous_deserialization=True
        )
    else:
        # faiss_db가 없으면 새로 생성
        splits = process_pdf()

        faiss_db = FAISS.from_documents(
            documents=splits,
            embedding=embedding_function
        )

        faiss_db.save_local(faiss_directory)

    return faiss_db

# 최근 4번의 대화만 chat_history로 chain에 입력되도록 수정
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    history = store[session_id]

    # 최근 4개 메시지만 유지
    if len(history.messages) > 4:
        history.messages = history.messages[-4:]

    return history

# 체인 초기화
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.

컨텍스트: {context}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 체인 초기화
        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()