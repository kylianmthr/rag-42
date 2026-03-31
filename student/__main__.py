import time
from typing import Optional
import fire
import re
import bm25s
import os
from transformers import GenerationConfig, pipeline
import json
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic_core import from_json

from student.validator import MinimalSource


class CLI:
    def index(
        self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000
    ) -> None:
        src = []
        python_content = []
        python_metadata = []
        md_content = []
        md_metadata = []
        for root, dirs, files in os.walk(path):
            for file in files:
                extension = os.path.splitext(file)[1]
                path_file = os.path.join(root, file)
                if extension in [".py", ".md"]:
                    with open(path_file, "r") as f:
                        content = f.read()
                        if extension == ".md":
                            md_content.append(content)
                            md_metadata.append({"source": path_file})
                        else:
                            python_content.append(content)
                            python_metadata.append({"source": path_file})
        splitter = MarkdownTextSplitter(
            chunk_size=max_chunk_size,
            add_start_index=True,
        )
        docs = splitter.create_documents(md_content, metadatas=md_metadata)
        for i, doc in enumerate(docs):
            src.append(
                MinimalSource(
                    file_path=doc.metadata["source"],
                    first_character_index=doc.metadata["start_index"],
                    last_character_index=doc.metadata["start_index"]
                    + len(doc.page_content),
                    page_content=doc.page_content,
                )
            )

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            add_start_index=True,
        )
        docs = splitter.create_documents(
            python_content, metadatas=python_metadata
        )
        for i, doc in enumerate(docs):
            src.append(
                MinimalSource(
                    file_path=doc.metadata["source"],
                    first_character_index=doc.metadata["start_index"],
                    last_character_index=doc.metadata["start_index"]
                    + len(doc.page_content),
                    page_content=doc.page_content,
                )
            )
        tokenized_content = [doc.page_content.split(" ") for doc in src]
        retriever = bm25s.BM25(corpus=tokenized_content)
        retriever.index(tokenized_content)
        retriever.save(
            "data/processed/bm25_index",
            corpus=[obj.model_dump() for obj in src],
        )
        chunk_data = [json.loads(doc.model_dump_json()) for doc in src]
        if not os.path.isdir("data/processed/chunks"):
            os.mkdir("data/processed/chunks")
        with open("data/processed/chunks/corpus.json", "w") as f:
            json.dump(chunk_data, f)

    def search(self, query: str, k: int):
        ret_loaded = bm25s.BM25.load(
            "data/processed/bm25_index", load_corpus=True
        )
        docs, scores = ret_loaded.retrieve(bm25s.tokenize(query), k=k)
        return docs

    def search_dataset(self):
        pass

    def answer(self, prompt: str, k: int = 2):
        docs = self.search(prompt, k)
        context = (
            "You are a technical assistant specializing in the vLLM codebase. Your goal is to provide accurate answers based ONLY on the provided context.\n"
            "### MANDATORY RULES ###\n"
            '1. GROUNDING: Use only the information from the "RETRIEVED_CONTEXT" section. If the answer is not present, state: "I\'m sorry, but the provided context does not contain enough information to answer this question."\n'
            "2. CITATIONS: You MUST cite the source file path for every claim or code snippet you provide. Use brackets at the end of the sentence, e.g., [vllm/engine/llm_engine.py].\n"
            "3. BREVITY: Keep your technical explanations concise and focus on the code implementation.)\n"
            "### RETRIEVED_CONTEXT ###\n"
        )
        for doc in docs[0]:
            context += f"# SOURCE: {doc['file_path']}\n"
            context += "---\n"
            context += doc["page_content"]
            context += "---\n"

        pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
        message = [
            {"role": "system", "content": context},
            {"role": "user", "content": "/no_think " + prompt},
        ]
        gen_cfg = GenerationConfig(
            max_new_tokens=50,
            do_sample=True,
        )
        start = time.time()
        print(
            re.sub(
                r"<think>.*?</think>",
                "",
                pipe(message, generation_config=gen_cfg)[0]["generated_text"][
                    2
                ]["content"],
                flags=re.DOTALL,
            ).strip(".")
        )
        print("Time:", time.time() - start)

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    fire.Fire(CLI)
