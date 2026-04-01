import torch


class Answer:
    @torch.inference_mode()
    def generate_answer(self, model, data):
        return model.generate(
            **data,
            max_new_tokens=50,
            do_sample=False,
            top_p=0.95,
            use_cache=True,
            # torch_dtype=torch.float16,
            # device_map="auto",
        )

    def decode(self, tokenizer, inputs, outputs):
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
