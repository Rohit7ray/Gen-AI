from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_classic.chains.sql_database.prompt import PROMPT_SUFFIX, _mssql_prompt


from few_shots import few_shots

import os
from dotenv import load_dotenv
load_dotenv()

#print(os.environ["api_key"])
print(os.environ["OPENAI_API_KEY"])

def get_few_shot_db_chain():
    llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
    )
# Database Connection
    db = SQLDatabase.from_uri(
    "mssql+pyodbc://@DESKTOP-QML3J7P\\SQLEXPRESS/atliq_tshirt?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes",
    include_tables=["t_shirts", "discounts"]
    )

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    _mssql_prompt = """
You are a Microsoft SQL Server (MSSQL) expert.
Given an input question, first create a syntactically correct MSSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies a specific number of examples to obtain, query for at most {top_k} results using the TOP clause as per MSSQL.
You can order the results to return the most informative data in the database.
Never query for all columns from a table. Query only the columns needed to answer the question.
Wrap column names in square brackets [] to denote them as delimited identifiers.
Use only the column names visible in the schema below. Be careful not to query columns that do not exist. Also pay attention to which column belongs to which table.
Use GETDATE() function if the question involves "today".
Do NOT use:
- LIMIT
- backticks `
- MySQL syntax

Use the following format:

Question: Question here
SQLQuery: Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mssql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain

