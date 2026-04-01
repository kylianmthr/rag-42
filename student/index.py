import os
from typing import Any, TypedDict
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
import bm25s
import json
from student.validator import MinimalSource


class ChunkDict(TypedDict):
    splitter: MarkdownTextSplitter | RecursiveCharacterTextSplitter
    content: list
    metadatas: list


class Index:
    def read_contents(
        self, path: str
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        contents = {"python_content": [], "md_content": []}
        metadatas = {"python_metadatas": [], "md_metadatas": []}
        for root, dirs, files in os.walk(path):
            for file in files:
                extension = os.path.splitext(file)[1]
                path_file = os.path.join(root, file)
                if extension in [".py", ".md"]:
                    with open(path_file, "r") as f:
                        content = f.read()
                        if extension == ".md":
                            contents["md_content"].append(content)
                            metadatas["md_metadatas"].append(
                                {"source": path_file}
                            )
                        else:
                            contents["python_content"].append(content)
                            metadatas["python_metadatas"].append(
                                {"source": path_file}
                            )
        return contents, metadatas

    def split(
        self,
        splitter: RecursiveCharacterTextSplitter | MarkdownTextSplitter,
        content: list,
        metadatas: list,
    ) -> list[MinimalSource]:
        src = []
        docs = splitter.create_documents(content, metadatas=metadatas)
        for i, doc in enumerate(docs):
            src.append(
                MinimalSource(
                    file_path=doc.metadata["source"],
                    first_character_index=doc.metadata["start_index"],
                    last_character_index=doc.metadata["start_index"]
                    + len(doc.page_content),
                    page_content=doc.page_content,
                )
            )
        return src

    def save(self, src: list[MinimalSource]):
        tokenized_content = [doc.page_content.split(" ") for doc in src]
        retriever = bm25s.BM25(corpus=tokenized_content)
        retriever.index(tokenized_content)
        retriever.save(
            "data/processed/bm25_index",
            corpus=[obj.model_dump() for obj in src],
        )
        chunk_data = [json.loads(doc.model_dump_json()) for doc in src]
        if not os.path.isdir("data/processed/chunks"):
            os.mkdir("data/processed/chunks")
        with open("data/processed/chunks/corpus.json", "w") as f:
            json.dump(chunk_data, f)
