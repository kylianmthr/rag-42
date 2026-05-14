import json
import os
from pathlib import Path
import re
from pydantic import BaseModel
from student.generate import Generate
from student.indexer import Indexer
import bm25s
import chromadb
from tqdm import tqdm

from student.validator import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


class RAG:
    def __init__(self) -> None:
        """Initialize retrieval, ranking, and generation components."""
        path = Path("data/processed/chunks")
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient("data/processed/chunks")
        self.rank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    def index(self, path: str, max_chunk_size: int) -> None:
        """Index a folder and build BM25 and vector stores.

        Args:
            path: Folder containing documents to index.
            max_chunk_size: Maximum chunk size for splitting.
        """
        indexer = Indexer(path, max_chunk_size)
        indexer.load_files()
        indexer.split()
        indexer.save(self.client)

    def load_index(self) -> bm25s.BM25:
        """Load the BM25 index from disk.

        Returns:
            Loaded BM25 retriever.
        """
        return bm25s.BM25.load("data/processed/bm25_index", load_corpus=True)

    def rerank(
        self, query: str, k: int, srcs: list[MinimalSource]
    ) -> list[MinimalSource]:
        """Rerank sources with a cross-encoder model.

        Args:
            query: User question or search query.
            k: Number of sources to keep.
            srcs: Candidate sources to score.

        Returns:
            Reranked list of sources.
        """
        srcs_with_score = {}
        for src in srcs:
            srcs_with_score[src.page_content] = self.rank_model.predict(
                [(query, src.page_content)]
            )
        srcs = sorted(
            srcs,
            key=lambda x: srcs_with_score[x.page_content][0],
            reverse=True,
        )
        return srcs[:k]

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Retrieve relevant sources for a query.

        Args:
            query: User question or search query.
            k: Number of sources to return.

        Returns:
            List of retrieved sources.
        """
        sources: list[MinimalSource] = []
        ret_loaded = self.load_index()
        collection = self.client.get_or_create_collection(
            name="chunks",
        )

        candidate_k = k

        docs, _ = ret_loaded.retrieve(bm25s.tokenize(query), k=candidate_k)
        sources += [MinimalSource(**doc) for doc in docs[0]]
        res = collection.query(query_texts=[query], n_results=candidate_k)
        if res["metadatas"]:
            sources += [
                MinimalSource(
                    **res["metadatas"][0][i],
                )
                for i in range(len(res["ids"][0]))
            ]

        unique_sources = []
        seen = set()
        for src in sources:
            if src.page_content not in seen:
                seen.add(src.page_content)
                unique_sources.append(src)

        return self.rerank(query, k, unique_sources)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
    ) -> None:
        """Run search for each dataset question and save results.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of sources to retrieve per question.
            save_directory: Output folder for search results.
        """
        search_results: list[MinimalSearchResults] = []
        with open(dataset_path, "r") as f:
            rag_dataset = RagDataset.model_validate_json(f.read())
        for _, question in enumerate(tqdm(rag_dataset.rag_questions)):
            search_res = self.search(question.question, k)
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=search_res,
                )
            )
        res = StudentSearchResults(search_results=search_results, k=k)
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.save_model(save_directory, "dataset_docs_public.json", res)

    def generate_pipeline(
        self, sources: list[MinimalSource], question: str, k: int
    ) -> str:
        """Generate an answer using retrieved sources and a question.

        Args:
            sources: Retrieved sources to use as context.
            question: Question to answer.
            k: Number of sources used.

        Returns:
            Final answer text.
        """
        generator = Generate(sources, question, k, self.model, self.tokenizer)
        inputs = generator.generate_inputs(
            generator.generate_context(), question
        )
        outputs = generator.generate_answer(inputs)
        answer = re.sub(
            r"<think>.*?</think>",
            "",
            generator.decode(inputs, outputs),
            flags=re.DOTALL,
        ).strip()
        return answer

    def answer(self, prompt: str, k: int) -> None:
        """Answer a single prompt and save the result to disk.

        Args:
            prompt: Question to answer.
            k: Number of sources to retrieve.
        """
        srcs = self.search(prompt, k)
        answer = self.generate_pipeline(srcs, prompt, k)
        if not os.path.isdir("data/output"):
            os.mkdir("data/output")
        with open("data/output/search_result.json", "w") as f:
            json.dump(
                json.loads(
                    self.generate_model(
                        answer, prompt, srcs, k
                    ).model_dump_json()
                ),
                f,
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers for a saved search results file.

        Args:
            student_search_results_path: Path to search results JSON.
            save_directory: Output folder for answers.
        """
        with open(student_search_results_path, "r") as f:
            searchs = StudentSearchResults(**json.loads(f.read()))
            answers: list[MinimalAnswer] = []
            for _, search in enumerate(tqdm(searchs.search_results)):
                answer = self.generate_pipeline(
                    search.retrieved_sources, search.question, searchs.k
                )
                answers.append(
                    MinimalAnswer(
                        question_id=search.question_id,
                        question=search.question,
                        retrieved_sources=search.retrieved_sources,
                        answer=answer,
                    )
                )
            self.save_model(
                path=save_directory,
                file="dataset_docs_public.json",
                model=StudentSearchResultsAndAnswer(
                    search_results=answers, k=searchs.k
                ),
            )

    def save_model(self, path: str, file: str, model: BaseModel) -> None:
        """Serialize a Pydantic model to JSON on disk.

        Args:
            path: Output folder.
            file: Output file name.
            model: Pydantic model to serialize.
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        with open(f"{path}/{file}", "w") as f:
            json.dump(json.loads(model.model_dump_json()), f, indent=4)

    def generate_model(
        self, answer: str, prompt: str, docs: list[MinimalSource], k: int
    ) -> StudentSearchResultsAndAnswer:
        """Build a single-entry result model from generated output.

        Args:
            answer: Generated answer text.
            prompt: Original question.
            docs: Retrieved sources used as context.
            k: Number of sources used.

        Returns:
            Structured result model for serialization.
        """
        return StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id=UnansweredQuestion(
                        question=prompt
                    ).question_id,
                    question=prompt,
                    retrieved_sources=docs,
                    answer=answer,
                )
            ],
            k=k,
        )
