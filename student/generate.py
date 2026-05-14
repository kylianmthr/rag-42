from typing import Any
import torch

from student.validator import (
    MinimalSource,
)


class Generate:
    def __init__(
        self,
        docs: list[MinimalSource],
        prompt: str,
        k: int,
        model: Any,
        tokenizer: Any,
    ) -> None:
        """Initialize a generator for answers.

        Args:
            docs: Retrieved sources used as context.
            prompt: User question to answer.
            k: Number of sources used.
            model: Causal language model used for generation.
            tokenizer: Tokenizer compatible with the model.
        """
        self.docs: list[MinimalSource] = docs
        self.context: None | str = None
        self.tokenizer = tokenizer
        self.model = model
        self.prompt = prompt
        self.k = k

    def generate_context(self) -> str:
        """Build the context block from retrieved sources.

        Returns:
            Context string containing concatenated sources.
        """
        context = "### RETRIEVED_CONTEXT ###\n"
        for doc in self.docs:
            context += f"# SOURCE: {doc.file_path}\n"
            context += "---\n"
            context += str(doc.page_content)
            context += "---\n"
        return context

    def generate_inputs(self, context: str, prompt: str) -> Any:
        """Create model inputs for a chat-style prompt.

        Args:
            context: Retrieved context for the system message.
            prompt: User question text.

        Returns:
            Tokenized inputs ready for model generation.
        """
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

    @torch.inference_mode()
    def generate_answer(self, data: Any) -> Any:
        """Run the model generation step.

        Args:
            data: Tokenized inputs from `generate_inputs`.

        Returns:
            Raw model outputs.
        """
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
        """Decode generated tokens into text.

        Args:
            inputs: Tokenized inputs used for generation.
            outputs: Raw generated token outputs.

        Returns:
            Decoded text string.
        """
        index = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0][index:])

    def limit(self, string: str) -> str:
        """Limit a string to a maximum length.

        Args:
            string: Input string.

        Returns:
            Truncated string when exceeding 400 characters.
        """
        if len(string) > 400:
            return string[:400]
        return string
