import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 RAG 챗봇",
    page_icon="📚",
    layout="wide"
)

# 초기 상태 설정
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# CSS 스타일
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important; /* 분홍색 */
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important; /* 노랑색 */
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important; /* 청색 */
}
h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
h5 {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 리스트 스타일 */
.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}

/* 강조 텍스트 스타일 */
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 인용문 스타일 */
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

/* 코드 스타일 */
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

/* 전체 텍스트 일관성 */
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 버튼 스타일 */
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    background-color: #ff1493 !important;
}
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown("""
<div style="text-align: center; margin-top: -4rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: 2.5rem; font-weight: bold; margin: 0;">
        <span style="color: #1f77b4;">PDF</span> 
        <span style="color: #ffffff; font-size: 0.7em;">기반</span> 
        <span style="color: #ffd700;">RAG</span> 
        <span style="color: #d62728; font-size: 0.7em;">챗봇</span>
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("PDF 파일을 업로드하고 내용에 관해 질문해보세요!")

# 사이드바 설정
with st.sidebar:
    st.markdown('<h2 style="color: #1f77b4;">API 키 설정</h2>', unsafe_allow_html=True)
    openai_api_key = st.text_input(
        "OpenAI API 키를 입력하세요",
        type="password",
        help="OpenAI API 키를 입력하세요. 이 키는 브라우저에 저장되지 않습니다."
    )
    
    # API 키가 입력되었는지 확인
    if not openai_api_key:
        st.warning("⚠️ OpenAI API 키를 입력해주세요.")
        st.stop()
    
    # API 키를 환경 변수로 설정
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.markdown('<h2 style="color: #1f77b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        process_button = st.button("파일 처리하기")
        
        if process_button:
            with st.spinner("PDF 파일을 처리 중입니다..."):
                try:
                    # 임시 파일 생성 및 처리
                    temp_dir = tempfile.TemporaryDirectory()
                    
                    all_docs = []
                    new_files = []
                    
                    # 각 파일 처리
                    for uploaded_file in uploaded_files:
                        # 이미 처리된 파일 스킵
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                            
                        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                        
                        # 업로드된 파일을 임시 파일로 저장
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # PDF 로더 생성 및 문서 로드
                        loader = PyPDFLoader(temp_file_path)
                        documents = loader.load()
                        
                        # 메타데이터에 파일 이름 추가
                        for doc in documents:
                            doc.metadata["source"] = uploaded_file.name
                        
                        all_docs.extend(documents)
                        new_files.append(uploaded_file.name)
                
                    if not all_docs:
                        st.success("모든 파일이 이미 처리되었습니다.")
                    else:
                        # 텍스트 분할
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100,
                            length_function=len
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        
                        # 모든 청크를 벡터 데이터베이스에 저장
                        total_chunks = len(chunks)
                        st.info(f"총 {total_chunks}개의 청크를 처리합니다.")
                        
                        # 임베딩 및 벡터 스토어 생성
                        embeddings = OpenAIEmbeddings()
                        
                        if st.session_state.vectorstore is None:
                            # 새 벡터 스토어 생성
                            batch_size = 30
                            vectorstore = None
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    if vectorstore is None:
                                        vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                                    else:
                                        vectorstore.add_documents(batch_chunks)
                                except Exception as e:
                                    continue
                            
                            st.session_state.vectorstore = vectorstore
                        else:
                            # 기존 벡터 스토어에 추가
                            batch_size = 30
                            
                            for i in range(0, len(chunks), batch_size):
                                batch_chunks = chunks[i:i + batch_size]
                                
                                try:
                                    st.session_state.vectorstore.add_documents(batch_chunks)
                                except Exception as e:
                                    continue
                        
                        # 검색기 생성 (더 많은 결과와 정확한 검색)
                        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 10}  # 검색 결과 수 증가
                        )
                        
                        # 처리된 파일 목록 업데이트
                        st.session_state.processed_files.extend(new_files)
                        
                except Exception as e:
                    st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                    st.error("파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")

    # 처리된 파일 목록 표시
    if st.session_state.processed_files:
        st.markdown('<h3 style="color: #ffd700;">처리된 파일 목록</h3>', unsafe_allow_html=True)
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    # 대화 초기화 버튼
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.rerun()
    
    # 메모리 사용량 표시
    if st.session_state.processed_files:
        st.subheader("📊 시스템 상태")
        st.info(f"처리된 파일 수: {len(st.session_state.processed_files)}")
        st.info(f"대화 기록 수: {len(st.session_state.chat_history)}")

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.write("먼저 PDF 파일을 업로드하고 처리해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 PDF 파일을 업로드하고 처리해주세요."})
    else:
        with st.spinner("답변을 생성 중입니다..."):
            try:
                # RAG 검색 (상위 3개 문서만 사용)
                retrieved_docs = st.session_state.retriever.invoke(prompt)
                
                if not retrieved_docs:
                    response = f"죄송합니다. '{prompt}'에 대한 관련 문서를 찾을 수 없습니다."
                else:
                    # 상위 3개 문서만 사용
                    top_docs = retrieved_docs[:3]
                    
                    # 컨텍스트 구성
                    context_text = ""
                    max_context_length = 8000
                    current_length = 0
                    
                    for i, doc in enumerate(top_docs):
                        doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                        if current_length + len(doc_text) > max_context_length:
                            st.warning(f"토큰 제한으로 인해 문서 {i+1}개만 사용합니다.")
                            break
                        context_text += doc_text
                        current_length += len(doc_text)
                    
                    # 과거 대화 맥락 구성
                    conversation_context = ""
                    if st.session_state.conversation_memory:
                        conversation_context = "\n\n=== 이전 대화 맥락 ===\n"
                        # 최근 50개 대화 사용
                        recent_conversations = st.session_state.conversation_memory[-50:]
                        for conv in recent_conversations:
                            conversation_context += f"{conv}\n"
                        conversation_context += "=== 대화 맥락 끝 ===\n"
                    
                    # 시스템 프롬프트 구성
                    system_prompt = f"""
                    질문: {prompt}
                    
                    관련 문서:
                    {context_text}{conversation_context}
                    
                    위 문서 내용과 이전 대화 맥락을 모두 고려하여 질문에 답변해주세요.
                    이전 대화에서 언급된 내용이 있다면 그것을 참조하여 더 정확하고 맥락적인 답변을 제공하세요.
                    
                    답변 형식:
                    - 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
                    - 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
                    - 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
                    - 답변은 서술형으로 작성하되 존대말을 사용하세요
                    - 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요
                    
                    주의사항:
                    - 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
                    - "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
                    - 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
                    - 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
                    """
                    
                    # LLM으로 답변 생성
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
                    response = llm.invoke(system_prompt).content
                    
                
                # 답변 표시
                with st.chat_message("assistant"):
                    st.write(response)
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # 대화 맥락 메모리에 추가 (최근 50개 대화 유지)
                st.session_state.conversation_memory.append(f"사용자: {prompt}")
                st.session_state.conversation_memory.append(f"AI: {response}")
                if len(st.session_state.conversation_memory) > 100:  # 50개 대화 = 100개 메시지
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                
            except Exception as e:
                with st.chat_message("assistant"):
                    st.write(f"오류가 발생했습니다: {str(e)}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
