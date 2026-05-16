import os
import time
import pickle
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# LCEL Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Streamlit UI
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

# URL Inputs
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=500,
    model="gpt-3.5-turbo"
)

# Process URLs
if process_url_clicked:

    # Remove empty URLs
    # urls = [url for url in urls if url.strip()]

    if not urls:
        st.error("Please enter at least one URL.")
        st.stop()

    # Load data
    main_placeholder.text("Loading data from URLs...")

    loader = UnstructuredURLLoader(urls=urls)

    try:
        data = loader.load()

    except Exception as e:
        st.error(f"Error loading URLs: {e}")
        st.stop()

    # Split documents
    main_placeholder.text("Splitting text into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
        # ,chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    # Create embeddings
    main_placeholder.text("Creating embeddings...")

    embeddings = OpenAIEmbeddings()

    # Create vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    time.sleep(2)

    # Save vector store
    vectorstore.save_local("faiss_index")

    st.success("URLs processed successfully!")

# Question Input
query = st.text_input("Question:")

# Query Processing
if query:

    if not os.path.exists("faiss_index"):
        st.error("Please process URLs first.")
        st.stop()

    # Load embeddings
    embeddings = OpenAIEmbeddings()

    # Load vector store
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context.

        Context:
        {context}

        Question:
        {question}
        """
    )

    # Function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL Chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate Response
    with st.spinner("Searching for answer..."):

        response = chain.invoke(query)

    # Display Answer
    st.header("Answer")
    st.write(response)

    # Display Sources
    st.subheader("Sources")
    retrieved_docs = retriever.invoke(query)
    unique_sources = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source")
        if source and source not in unique_sources:
            unique_sources.add(source)
            st.write(source)

