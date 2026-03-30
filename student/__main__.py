from typing import Optional
import fire
import os
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from student.validator import MinimalSource


class CLI:
    def index(self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000):
        src = []
        python_content = []
        python_metadata = []
        md_content = []
        md_metadata = []
        for root, dirs, files in os.walk(path):
            for file in files:
                extension = os.path.splitext(file)[1]
                path_file = os.path.join(root, file)
                if extension in [".py", ".md"]:
                    with open(path_file, "r") as f:
                        content = f.read()
                        if extension == ".md":
                            md_content.append(content)
                            md_metadata.append({"source": path_file})
                        else:
                            python_content.append(content)
                            python_metadata.append({"source": path_file})
                            # splitter: Optional[
                            #    MarkdownTextSplitter
                            #    | RecursiveCharacterTextSplitter
                            # ] = None
                            # if extension == ".md":
                            #    splitter = MarkdownTextSplitter(
                            #        chunk_size=max_chunk_size,
                            #        add_start_index=True,
                            #    )
                            # else:
                            #    splitter = (
                            #        RecursiveCharacterTextSplitter.from_language(
                            #            language=Language.PYTHON,
                            #            chunk_size=max_chunk_size,
                            #            add_start_index=True,
                            #        )
                            #    )
                        # src.append(MinimalSource(path_file, splitter))

        splitter = MarkdownTextSplitter(
            chunk_size=max_chunk_size,
            add_start_index=True,
        )
        docs = splitter.create_documents(md_content, metadatas=md_metadata)
        for i, doc in enumerate(docs):
            src.append(
                MinimalSource(
                    file_path=doc.metadata["source"],
                    first_character_index=doc.metadata["start_index"],
                    last_character_index=doc.metadata["start_index"]
                    + len(doc.page_content),
                )
            )

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            add_start_index=True,
        )
        docs = splitter.create_documents(
            python_content, metadatas=python_metadata
        )
        for i, doc in enumerate(docs):
            src.append(
                MinimalSource(
                    file_path=doc.metadata["source"],
                    first_character_index=doc.metadata["start_index"],
                    last_character_index=doc.metadata["start_index"]
                    + len(doc.page_content),
                )
            )

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
