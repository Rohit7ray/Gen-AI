# AI SQL Data Assistant using LangChain & LLM

An AI-powered SQL Assistant that enables users to query a SQL Server database using natural language. Instead of writing SQL queries manually, users can simply ask questions in English, and the application automatically generates SQL, executes it against the database, and returns human-readable answers.

---

## Features

- Natural Language to SQL conversion
- SQL Server database integration
- Custom Prompt Engineering
- Few-Shot Learning
- Semantic Similarity Search
- Chroma Vector Database
- HuggingFace Embeddings
- Automatic SQL execution
- Business-friendly response generation
- Reduced LLM hallucinations using prompt constraints

---

## Tech Stack

- Python
- LangChain
- OpenAI GPT-3.5 Turbo
- SQL Server
- SQLDatabaseChain
- HuggingFace Embeddings
- Sentence Transformers
- Chroma Vector Database
- Prompt Engineering
- Few-Shot Prompting

---

## Project Workflow

User Question
↓
Prompt Engineering
↓
Semantic Similarity Search
↓
Relevant Few-Shot Examples
↓
LLM (GPT-3.5)
↓
SQL Query Generation
↓
SQL Server Execution
↓
Business-Friendly Response

---

## Key Concepts Implemented

- Large Language Models (LLMs)
- Prompt Engineering
- Few-Shot Learning
- Embeddings
- Vector Databases
- Retrieval-based Prompt Selection
- Natural Language Processing
- SQL Query Generation
- AI-powered Database Assistant

---

## Example Questions

- How many white Levi shirts do we have?
- What is the total inventory value for small-sized shirts?
- How much revenue will be generated after discounts?
- What is the available stock for Nike XS shirts?

---

## Challenges Solved

- Reduced SQL hallucinations using custom prompt templates.
- Improved SQL generation accuracy using semantic example retrieval.
- Dynamically selected the most relevant examples using Chroma Vector Database.
- Prevented invalid table and column generation through prompt constraints.

---


## Skills Demonstrated

- Python
- LangChain
- OpenAI APIs
- Prompt Engineering
- RAG Concepts
- Embeddings
- Chroma DB
- SQL Server
- Vector Search
- AI Application Development
