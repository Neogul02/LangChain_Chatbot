import streamlit as st # streamlit 라이브러리
import tiktoken # text를 토큰갯수로 변환하는 라이브러리

from loguru import logger # 로그 라이브러리

from langchain.chains import ConversationalRetrievalChain 
from langchain.chat_models import ChatOpenAI # OpenAI 모델

from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import Docx2txtLoader 
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter # 텍스트 분할 라이브러리
from langchain.embeddings import HuggingFaceEmbeddings # 임베딩 라이브러리

from langchain.memory import ConversationBufferMemory # 몇개까지의 메모리를 저장할지 결정하는 라이브러리
from langchain.vectorstores import FAISS # 벡터스토어 라이브러리 (임시로)

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback 
from langchain.memory import StreamlitChatMessageHistory


def main():
    st.set_page_config(  # 페이지 이름
        page_title="PDF Chatbot",)

    st.title("LangChain_Chat QA")  # 제목

    if "conversation" not in st.session_state:
        st.session_state.conversation = None 

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:  # 사이드바
        uploaded_files = st.file_uploader("파일 업로드", type=[
                                          'pdf', 'docx'], accept_multiple_files=True)
        openai_api_key = st.text_input( 
            "OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:  # API 키 확인
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)  # 파일 텍스트 추출
        text_chunks = get_text_chunks(files_text)  # 텍스트 청크로 분할
        vetorestore = get_vectorstore(text_chunks)  # 벡터화

        st.session_state.conversation = get_conversation_chain(
            vetorestore, openai_api_key)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:  # 채팅 시작 메시지
        st.session_state['messages'] = [{"role": "assistant",
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # 채팅 메시지 아이콘
            st.markdown(message["content"]) # 채팅 메시지 내용

    history = StreamlitChatMessageHistory(key="chat_messages")

    #  Chat logic
    if query := st.chat_input("질문을 입력해주세요."): # 사용자가 질문을 입력하면 ~ 
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"): # 사용자가 입력한 질문을 출력
            st.markdown(query)

        with st.chat_message("assistant"): # 챗봇이 답변을 출력

            chain = st.session_state.conversation # get_conversation_chain 함수에서 반환된 값

            with st.spinner("Thinking..."): # 로딩 
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"): # 펼치기/접기 가능한 문서 목록
                    st.markdown(
                        source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    st.markdown(
                        source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    st.markdown(
                        source_documents[2].metadata['source'], help=source_documents[2].page_content)

# Add assistant message to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


def tiktoken_len(text): # 토큰 갯수를 기준으로 텍스트를 스플릿
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs): # 업로드 된 파일에서 텍스트 추출

    doc_list = []  # 문서 리스트, 여러개의 파일

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}") # 로그에 파일 이름 출력

        if '.pdf' in doc.name:  # 파일이 pdf일 경우
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:  # 파일이 docx일 경우
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:  # 파일이 pptx일 경우
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,  # 900자로 스플릿
        chunk_overlap=100, 
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text) # 문서를 스플릿
    return chunks


def get_vectorstore(text_chunks): # 벡터스토어 생성
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings) 
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     model_name='gpt-3.5-turbo', temperature=0) # OpenAI 모델
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(
            memory_key='chat_history', return_messages=True, output_key='answer'), # 이전 대화를 기억 
        get_chat_history=lambda h: h, 
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain


if __name__ == '__main__':
    main()
