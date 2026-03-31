from typing import TypedDict
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from student.validator import MinimalSource


class ChunkDict(TypedDict):
    splitter: MarkdownTextSplitter | RecursiveCharacterTextSplitter
    content: list
    metadatas: list


class Index:
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
