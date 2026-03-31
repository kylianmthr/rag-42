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

from student.index import ChunkDict, Index
from student.validator import MinimalSource


class CLI:
    def index(
        self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000
    ) -> None:
        contents = {"python_content": [], "md_content": []}
        metadatas = {"python_metadatas": [], "md_metadatas": []}
        src = []
        for root, dirs, files in os.walk(path):
            for file in files:
                extension = os.path.splitext(file)[1]
                path_file = os.path.join(root, file)
                if extension in [".py", ".md"]:
                    with open(path_file, "r") as f:
                        content = f.read()
                        if extension == ".md":
                            contents["md_content"].append(content)
                            metadatas["md_metadatas"].append(
                                {"source": path_file}
                            )
                        else:
                            contents["python_content"].append(content)
                            metadatas["python_metadatas"].append(
                                {"source": path_file}
                            )
        chunks: list[ChunkDict] = [
            {
                "splitter": MarkdownTextSplitter(
                    chunk_size=max_chunk_size,
                    add_start_index=True,
                ),
                "content": contents["md_content"],
                "metadatas": metadatas["md_metadatas"],
            },
            {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=max_chunk_size,
                    add_start_index=True,
                ),
                "content": contents["python_content"],
                "metadatas": metadatas["python_metadatas"],
            },
        ]
        for chunk in chunks:
            src += Index().split(
                chunk["splitter"], chunk["content"], chunk["metadatas"]
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
