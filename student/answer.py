import torch


class Answer:
    @torch.inference_mode()
    def generate_answer(self, model, data):
        return model.generate(
            **data,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            cache_implementation="static",
            # torch_dtype=torch.float16,
            # device_map="auto",
        )

    def decode(self, tokenizer, inputs, outputs):
        index = inputs["input_ids"].shape[-1]
        return tokenizer.decode(outputs[0][index:])

    def limit(self, string: str):
        if len(string) > 400:
            return string[:400]
        return string
