import os
import streamlit as st
import time
import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

os.environ['OPENAI_API_KEY']  = 'API'

#creating LLM model
llm =ChatOpenAI(model = 'gpt-4o-mini', temperature = 0.9, max_tokens = 500)

##--------------1.LOAD--------------

loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
])
data = loaders.load()


##--------------2. SPLITTER--------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)



##--------------3. EMBEDDING FAISS INDEX--------------

# Create the embeddings of the chunks using openAIEmbeddings
embeddings = OpenAIEmbeddings()
# Pass the documents and embeddings inorder to create FAISS vector index
vectorindex_openai = FAISS.from_documents(docs, embeddings)


# Save
vectorindex_openai.save_local("faiss_index")
# Load
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)



##--------------4. RETRIEVE--------------

retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_template(
    "Answer based only on the context:\n{context}\n\nQuestion: {input}"
)
# LCEL chain (no langchain.chains needed)
chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
)

response = chain.invoke("What is this article about?")
print(response.content)


