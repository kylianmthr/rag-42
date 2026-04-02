from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from student.validator import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


class Generate:
    def __init__(self, docs: list[MinimalSource], prompt: str, k: int) -> None:
        self.docs: list[MinimalSource] = docs
        self.context: None | str = None
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
        self.prompt = prompt
        self.k = k

    def generate_context(self) -> str:
        context = "### RETRIEVED_CONTEXT ###\n"
        for doc in self.docs:
            context += f"# SOURCE: {doc.file_path}\n"
            context += "---\n"
            context += doc.page_content
            context += "---\n"
        return context

    def generate_inputs(self, context: str, prompt: str) -> Any:
        messages = [
            {"role": "system", "content": self.limit(context)},
            {"role": "user", "content": "/no_think " + prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs

    @torch.inference_mode()  # type: ignore
    def generate_answer(self, data: Any) -> Any:
        return self.model.generate(
            **data,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
            # torch_dtype=torch.float16,
            # device_map="auto",
        )

    def decode(self, inputs: Any, outputs: Any) -> Any:
        index = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0][index:])

    def limit(self, string: str) -> str:
        if len(string) > 400:
            return string[:400]
        return string

    def generate_model(self, answer: str) -> StudentSearchResultsAndAnswer:
        return StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=UnansweredQuestion(
                        question=self.prompt
                    ).question_id,
                    question=self.prompt,
                    retrieved_sources=self.docs,
                    answer=answer,
                )
            ],
            k=self.k,
        )
