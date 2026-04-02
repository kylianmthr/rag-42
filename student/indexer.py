import os
import uuid
import chromadb
import dotenv
from chromadb.utils import embedding_functions
from typing import Any, TypedDict
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
import bm25s
import json
from student.errors import EmptyFolder
from student.validator import MinimalSource


class ChunkDict(TypedDict):
    splitter: MarkdownTextSplitter | RecursiveCharacterTextSplitter
    content: list[Any]
    metadatas: list[Any]


class Indexer:
    def __init__(self, path: str, max_chunk_size: int) -> None:
        self.contents: dict[str, list[Any]] = {
            "python_content": [],
            "md_content": [],
        }
        self.metadatas: dict[str, list[Any]] = {
            "python_metadatas": [],
            "md_metadatas": [],
        }
        self.ids: list[str] = []
        self.src: list[MinimalSource] = []
        self.max_chunk_size = max_chunk_size
        self.path = path

    def load_files(self) -> None:
        for root, dirs, files in os.walk(self.path):
            for file in files:
                extension = os.path.splitext(file)[1]
                path_file = os.path.join(root, file)
                if extension in [".py", ".md"]:
                    with open(path_file, "r") as f:
                        content = f.read()
                        if extension == ".md":
                            self.contents["md_content"].append(content)
                            self.metadatas["md_metadatas"].append(
                                {"source": path_file}
                            )
                        else:
                            self.contents["python_content"].append(content)
                            self.metadatas["python_metadatas"].append(
                                {"source": path_file}
                            )
                        self.ids.append(str(uuid.uuid4()))
        if not len(self.contents["python_content"]) and not len(
            self.contents["md_content"]
        ):
            raise EmptyFolder("The folder doesn't contain .md or .py file")

    def specific_split(
        self,
        splitter: RecursiveCharacterTextSplitter | MarkdownTextSplitter,
        content: list[Any],
        metadatas: list[Any],
    ) -> None:
        docs = splitter.create_documents(content, metadatas=metadatas)
        for i, doc in enumerate(docs):
            self.src.append(
                MinimalSource(
                    file_path=doc.metadata["source"],
                    first_character_index=doc.metadata["start_index"],
                    last_character_index=doc.metadata["start_index"]
                    + len(doc.page_content),
                    page_content=doc.page_content,
                )
            )

    def split(self) -> None:
        chunks: list[ChunkDict] = [
            {
                "splitter": MarkdownTextSplitter(
                    chunk_size=self.max_chunk_size,
                    add_start_index=True,
                ),
                "content": self.contents["md_content"],
                "metadatas": self.metadatas["md_metadatas"],
            },
            {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=self.max_chunk_size,
                    add_start_index=True,
                ),
                "content": self.contents["python_content"],
                "metadatas": self.metadatas["python_metadatas"],
            },
        ]
        for chunk in chunks:
            self.specific_split(
                chunk["splitter"], chunk["content"], chunk["metadatas"]
            )

    def save(self) -> None:
        tokenized_content = [doc.page_content.split(" ") for doc in self.src]
        retriever = bm25s.BM25(corpus=tokenized_content)
        retriever.index(tokenized_content)
        retriever.save(
            "data/processed/bm25_index",
            corpus=[obj.model_dump() for obj in self.src],
        )
        if not os.path.isdir("data/processed/chunks"):
            os.mkdir("data/processed/chunks")
        client = chromadb.PersistentClient(path="data/processed/chunks")
        dotenv.load_dotenv()
        collection = client.get_or_create_collection(
            name="chunks",
        )
        collection.add(
            documents=self.contents["python_content"]
            + self.contents["md_content"],
            metadatas=self.metadatas["python_metadatas"]
            + self.metadatas["md_metadatas"],
            ids=self.ids,
        )
        # chunk_data = [json.loads(doc.model_dump_json()) for doc in self.src]
        # if not os.path.isdir("data/processed/chunks"):
        #    os.mkdir("data/processed/chunks")
        # with open("data/processed/chunks/corpus.json", "w") as f:
        #    json.dump(chunk_data, f)
