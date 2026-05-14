*This project has been created as part of the 42 curriculum by kmathuri.* 

---

# Description

This project focuses on building a **Retrieval-Augmented Generation (RAG)** system designed to interface with the **vLLM** repository. The goal is to create a searchable knowledge base from source code and documentation to provide accurate, evidence-based answers to technical questions using a local LLM. 

# System Architecture

The pipeline is designed for high-precision retrieval and grounded generation:

1. 
**Ingestion Layer**: Processes Python and Markdown files from the vLLM repository. 


2. 
**Indexing Layer**: Implements a dual-indexing strategy using **BM25s** for lexical search and **ChromaDB** for vector storage. 


3. **Hybrid Retrieval**: Combines keyword matching with semantic embeddings (`ms-marco-MiniLM-L-6-v2`).
4. 
**Reranking**: Utilizes a cross-encoder to refine the top-k results, ensuring the most relevant context is passed to the model. 


5. 
**Generation**: Employs **Qwen3-0.6B** to produce structured JSON responses based on the retrieved context. 



# Chunking Strategy

We implement specialized chunking to maintain the structural integrity of the data: 

* 
**Python Code**: Leverages `bm25s` splitters to isolate logical blocks (classes and functions) while preserving scope. 


* 
**Documentation**: Markdown files are segmented by headers and paragraphs to ensure semantic coherence. 


* 
**Constraints**: Chunks are limited to a maximum of **2000 characters**, configurable via CLI arguments. 



# Retrieval Method

The system utilizes a **Hybrid Retrieval** mechanism: 

* 
**Lexical Search (BM25)**: Essential for locating specific technical identifiers, function names, and unique syntax within the codebase. 


* 
**Semantic Search (Dense Embeddings)**: Captures the conceptual meaning of queries, identifying relevant documentation even when keywords do not perfectly match. 


* 
**Cross-Encoding**: After the initial retrieval, a reranking step evaluates the deep semantic relationship between the query and each chunk to maximize the quality of the context provided to the LLM. 



# Performance Analysis

The implementation meets or exceeds the mandatory performance thresholds: 

* 
**Recall@5 (Docs)**: **~83%** (Required: 80%). 


* 
**Recall@5 (Code)**: **~63%** (Required: 50%). 


* 
**Indexing Time**: **3:00 - 4:30 minutes** (Required: < 5:00 minutes). 


* 
**Latency**: Cold start and warm retrieval times remain well within the 60s and 90s limits respectively. 



# Design Decisions

* 
**Pydantic Validation**: All data structures are enforced via Pydantic models to ensure strict schema adherence and prevent runtime failures. 


* 
**Package Management**: The project uses `uv` for lightning-fast dependency resolution and consistent environments. 


* 
**CLI with Python Fire**: A robust command-line interface provides clean access to every stage of the pipeline. 



# Challenges Faced

The primary challenge involved optimizing the **Recall@k** for source code. Unlike natural language, code requires a high degree of precision regarding variable names and logic flow. Balancing the weights between the BM25 lexical scores and the semantic vector scores was critical to achieving the required 50% recall on code-specific questions. 

# Example Usage

## Installation

```bash
make install

```

## Commands

```bash
# 1. Index the repository
uv run python -m student index --max_chunk_size 2000

# 2. Search for a single query
uv run python -m student search "How to implement a new model?" --k 5

# 3. Answer a question with context
uv run python -m student answer "What is the default block size in vLLM?" --k 10

```

# Resources

* [vLLM Documentation](https://docs.vllm.ai/en/latest/)
* [LangChain Documentation](https://python.langchain.com/)
* [BM25s Performance Metrics](https://github.com/xhluca/bm25s)

## AI Usage Disclosure

AI tools were utilized for the following tasks: 

* Generating initial structure and content for the `README.md`. 


* Assisting with complex Python type hinting and Pydantic model configurations. 


* Drafting comprehensive Google-style docstrings for internal functions. 
