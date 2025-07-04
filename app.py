## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
import time
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import hashlib
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()

# Utility: hash uploaded files
def hash_uploaded_files(files):
    hasher = hashlib.sha256()
    for f in files:
        hasher.update(f.name.encode())
        hasher.update(f.getvalue())
    return hasher.hexdigest()


# Streamlit UI
st.title("Conversational RAG With PDF uploads and chat history")
st.markdown("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        current_hash = hash_uploaded_files(uploaded_files)

        # Check if vectorstore should be updated
        if st.session_state.get("last_uploaded_hash") != current_hash:
            st.session_state.last_uploaded_hash = current_hash

            documents = []
            for uploaded_file in uploaded_files:
                file_path = f"./temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever()
            st.success("PDFs processed and vector store updated.")
        else:
            st.info("Same PDFs detected — using cached vector store.")

        # Use retriever from session
        retriever = st.session_state.retriever

        # Contextual question reformulation
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session-based memory
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            start=time.process_time()
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            stop=time.process_time()
            st.write(f"⏱️ Time taken: {stop - start:.2f} seconds")
            st.write("Assistant:", response['answer'])
            with st.expander("Chat History"):
                st.write("Chat History:", session_history.messages)
    else:
        st.info("Please upload at least one PDF to begin.")
else:
    st.warning("Please enter the Groq API Key.")