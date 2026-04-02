import json
import os
import re
from numpy._typing import NDArray
from pydantic import BaseModel
from student.generate import Generate
from student.indexer import Indexer
import bm25s

from student.validator import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class RAG:
    def index(self, path: str, max_chunk_size: int):
        indexer = Indexer(path, max_chunk_size)
        indexer.load_files()
        indexer.split()
        indexer.save()

    def load_index(self) -> bm25s.BM25:
        return bm25s.BM25.load("data/processed/bm25_index", load_corpus=True)

    def search(self, query: str, k: int) -> NDArray:
        ret_loaded = self.load_index()
        docs, scores = ret_loaded.retrieve(bm25s.tokenize(query), k=k)
        return docs

    def search_dataset(
        self,
        dataset_path,
        k,
        save_directory,
    ):
        search_results: list[MinimalSearchResults] = []
        with open(dataset_path, "r") as f:
            rag_dataset = RagDataset.model_validate_json(f.read())
        for question in rag_dataset.rag_questions:
            search_res = self.search(question.question, k)
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[
                        MinimalSource(**doc) for doc in search_res[0]
                    ],
                )
            )
        res = StudentSearchResults(search_results=search_results, k=k)
        self.save_model(save_directory, "search_result.json", res)

    def answer(self, prompt, k):
        generator = Generate(
            [MinimalSource(**doc) for doc in self.search(prompt, k)[0]],
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

    def save_model(self, path: str, file: str, model: BaseModel):
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(f"{path}/{file}", "w") as f:
            json.dump(json.loads(model.model_dump_json()), f, indent=4)
