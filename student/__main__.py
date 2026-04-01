import time
import fire
import re
import bm25s
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from student.index import ChunkDict, Index
from student.answer import Answer


class CLI:
    def index(
        self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000
    ) -> None:
        contents, metadatas = Index().read_contents(path)
        src = []
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
        Index().save(src)

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
        context = "### RETRIEVED_CONTEXT ###\n"
        for doc in docs[0]:
            context += f"# SOURCE: {doc['file_path']}\n"
            context += "---\n"
            context += doc["page_content"]
            context += "---\n"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        # pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B", device=0)
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
        # print(Answer().decode(tokenizer, inputs, outputs))
        # gen_cfg = GenerationConfig(
        #    max_new_tokens=50,
        #    do_sample=True,
        # )
        # print(pipe(messages, generation_config=gen_cfg))
        print("Time:", time.time() - start)
        print(
            re.sub(
                r"<think>.*?</think>",
                "",
                Answer().decode(tokenizer, inputs, outputs),
                # pipe(messages),
                flags=re.DOTALL,
            ).strip(".")
        )

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    fire.Fire(CLI)
