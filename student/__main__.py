import json
import os
import time
import fire
import re
import bm25s
from numpy._typing import NDArray
from student.indexer import Indexer
from student.validator import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from student.answer import Answer


class CLI:
    def index(
        self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000
    ) -> None:
        indexer = Indexer(path, max_chunk_size)
        indexer.load_files()
        indexer.split()
        indexer.save()

    def search(self, query: str, k: int) -> NDArray:
        ret_loaded = bm25s.BM25.load(
            "data/processed/bm25_index", load_corpus=True
        )
        docs, scores = ret_loaded.retrieve(bm25s.tokenize(query), k=k)
        return docs

    def search_dataset(
        self,
        dataset_path: str = "datasets_public/public/UnansweredQuestions/dataset_code_public.json",
        k: int = 1,
        save_directory: str = "test",
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
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)
        with open(f"{save_directory}/output.json", "w") as f:
            json.dump(json.loads(res.model_dump_json()), f)

    def answer(self, prompt: str, k: int = 2):
        unanswered = UnansweredQuestion(question=prompt)
        docs = self.search(prompt, k)
        context = "### RETRIEVED_CONTEXT ###\n"
        for doc in docs[0]:
            context += f"# SOURCE: {doc['file_path']}\n"
            context += "---\n"
            context += doc["page_content"]
            context += "---\n"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        messages = [
            {"role": "system", "content": Answer().limit(context)},
            {"role": "user", "content": "/no_think " + prompt},
        ]
        print(len(Answer().limit(prompt)))
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        start = time.time()
        outputs = Answer().generate_answer(model, inputs)
        print("Time:", time.time() - start)
        answer = re.sub(
            r"<think>.*?</think>",
            "",
            Answer().decode(tokenizer, inputs, outputs),
            flags=re.DOTALL,
        ).strip()
        res = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=unanswered.question_id,
                    question=prompt,
                    retrieved_sources=[
                        MinimalSource(**doc) for doc in docs[0]
                    ],
                    answer=answer,
                )
            ],
            k=k,
        )
        res_json = json.loads(res.model_dump_json())
        if not os.path.isdir("data/output"):
            os.mkdir("data/output")
        with open("data/output/search_result.json", "w") as f:
            json.dump(res_json, f)

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    fire.Fire(CLI)
