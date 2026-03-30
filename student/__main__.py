from typing import Optional
import fire
import os
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)


class CLI:
    def index(self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000):
        for root, dirs, files in os.walk(path):
            for file in files:
                extension = os.path.splitext(file)[1]
                if extension in [".py", ".md"]:
                    with open(os.path.join(root, file), "r") as f:
                        content = f.read()
                        splitter: Optional[
                            MarkdownTextSplitter
                            | RecursiveCharacterTextSplitter
                        ] = None
                        if extension == ".md":
                            splitter = MarkdownTextSplitter(
                                chunk_size=max_chunk_size,
                            )
                        else:
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=max_chunk_size,
                            )
                        print(splitter.split_text(content))

        print(path)

    def search(self):
        pass

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    fire.Fire(CLI)
