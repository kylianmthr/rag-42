import json
import os
from pathlib import Path
import re
from typing import Any
from langchain_core import documents
from pydantic import BaseModel
from student.generate import Generate
from student.indexer import Indexer
import bm25s
import chromadb

from student.validator import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from sentence_transformers import CrossEncoder


class RAG:
    def __init__(self) -> None:
        path = Path("data/processed/chunks")
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient("data/processed/chunks")
        self.rank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def index(self, path: str, max_chunk_size: int) -> None:
        indexer = Indexer(path, max_chunk_size)
        indexer.load_files()
        indexer.split()
        indexer.save(self.client)

    def load_index(self) -> bm25s.BM25:
        return bm25s.BM25.load("data/processed/bm25_index", load_corpus=True)

    def rerank(
        self, query: str, k: int, srcs: list[MinimalSource]
    ) -> list[MinimalSource]:
        srcs_with_score = {}
        for src in srcs:
            srcs_with_score[src.page_content] = self.rank_model.predict(
                [(query, src.page_content)]
            )
        srcs = sorted(srcs, key=lambda x: srcs_with_score[x.page_content][0])
        return srcs[:k]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        sources: list[MinimalSource] = []
        ret_loaded = self.load_index()
        collection = self.client.get_or_create_collection(
            name="chunks",
        )
        docs, scores = ret_loaded.retrieve(bm25s.tokenize(query), k=k)
        sources += [MinimalSource(**doc) for doc in docs[0]]
        res = collection.query(query_texts=[query], n_results=k)
        if res["metadatas"]:
            sources += [
                MinimalSource(
                    **res["metadatas"][0][i],
                )
                for i in range(len(res["ids"][0]))
            ]
        return self.rerank(query, k, sources)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
    ) -> None:
        search_results: list[MinimalSearchResults] = []
        with open(dataset_path, "r") as f:
            rag_dataset = RagDataset.model_validate_json(f.read())
        for question in rag_dataset.rag_questions:
            search_res = self.search(question.question, k)
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=search_res,
                )
            )
        res = StudentSearchResults(search_results=search_results, k=k)
        self.save_model(save_directory, "search_result.json", res)

    def answer(self, prompt: str, k: int) -> None:
        generator = Generate(
            self.search(prompt, k),
            prompt,
            k,
        )
        inputs = generator.generate_inputs(
            generator.generate_context(), prompt
        )
        outputs = generator.generate_answer(inputs)
        answer = re.sub(
            r"<think>.*?</think>",
            "",
            generator.decode(inputs, outputs),
            flags=re.DOTALL,
        ).strip()
        if not os.path.isdir("data/output"):
            os.mkdir("data/output")
        with open("data/output/search_result.json", "w") as f:
            json.dump(
                json.loads(generator.generate_model(answer).model_dump_json()),
                f,
            )

    def save_model(self, path: str, file: str, model: BaseModel) -> None:
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(f"{path}/{file}", "w") as f:
            json.dump(json.loads(model.model_dump_json()), f, indent=4)
