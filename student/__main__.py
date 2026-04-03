import json
from typing import Optional
import fire
from student.rag import RAG
from student.validator import (
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class CLI:
    def __init__(self) -> None:
        self.rag = RAG()

    def index(
        self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000
    ) -> None:
        try:
            self.rag.index(path, max_chunk_size)
        except (FileNotFoundError, PermissionError) as e:
            print("[Error]: Error while trying to open the file/folder:", e)
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except Exception as e:
            print("[Error]:", e)

    def search(self, query: str, k: int) -> None:
        try:
            docs = self.rag.search(query, k)
            res = StudentSearchResults(
                search_results=[
                    MinimalSearchResults(
                        question_id="1",
                        question=query,
                        retrieved_sources=docs,
                    )
                ],
                k=k,
            )
            self.rag.save_model("data/output", "search_result.json", res)

        except (FileNotFoundError, PermissionError) as e:
            print("[Error]: Error while loading index:", e)
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except Exception as e:
            print("[Error]:", e)

    def search_dataset(
        self,
        dataset_path: str = (
            "datasets_public/public/"
            "UnansweredQuestions/dataset_code_public.json"
        ),
        k: int = 1,
        save_directory: str = "data/output/search_results",
    ) -> None:
        try:
            self.rag.search_dataset(dataset_path, k, save_directory)
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except Exception as e:
            print("[Error]:", e)

    def answer(self, prompt: str, k: int = 2) -> None:
        try:
            self.rag.answer(prompt, k)
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except Exception as e:
            print("[Error]:", e)

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/output/search_results/dataset_docs_public.json"
        ),
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        try:
            self.rag.answer_dataset(
                student_search_results_path, save_directory
            )
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except Exception as e:
            print("[Error]:", e)

    def inter(self, index_starts, index_ends):
        return max(0, min(index_ends) - max(index_starts)) / (
            index_ends[1] - index_starts[1]
        )

    def evaluate(
        self,
        student_answer_path: str = "data/output/search_results/dataset_docs_public.json",
        dataset_path: str = "datasets_public/public/AnsweredQuestions/dataset_docs_public.json",
    ) -> None:
        with open(student_answer_path, "r") as f:
            search_results = StudentSearchResults(**json.loads(f.read()))
        with open(dataset_path, "r") as f:
            dataset = RagDataset(**json.loads(f.read()))
        if search_results and dataset:
            for i in range(len(search_results.search_results)):
                recall = 0
                for j in range(search_results.k):
                    recall += self.inter(
                        [
                            search_results.search_results[i]
                            .retrieved_sources[j]
                            .first_character_index,
                            dataset.rag_questions[i]
                            .sources[j]
                            .first_character_index,
                        ],
                        [
                            search_results.search_results[i]
                            .retrieved_sources[j]
                            .last_character_index,
                            dataset.rag_questions[i]
                            .sources[j]
                            .last_character_index,
                        ],
                    )
                print(f"Recall@{i}:", recall / search_results.k)


if __name__ == "__main__":
    fire.Fire(CLI)
