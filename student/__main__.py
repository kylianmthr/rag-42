import fire
from student.rag import RAG
from student.validator import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


class CLI:
    def __init__(self) -> None:
        self.rag = RAG()

    def index(
        self, path: str = "vllm-0.10.1/", max_chunk_size: int = 2000
    ) -> None:
        self.rag.index(path, max_chunk_size)

    def search(self, query: str, k: int) -> None:
        docs = self.rag.search(query, k)
        print(docs[0])
        res = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="1",
                    question=query,
                    retrieved_sources=[
                        MinimalSource(**doc) for doc in docs[0]
                    ],
                )
            ],
            k=k,
        )
        self.rag.save_model("data/output", "search_result.json", res)

    def search_dataset(
        self,
        dataset_path: str = (
            "datasets_public/public/"
            "UnansweredQuestions/dataset_code_public.json"
        ),
        k: int = 1,
        save_directory: str = "data/output",
    ) -> None:
        self.rag.search_dataset(dataset_path, k, save_directory)

    def answer(self, prompt: str, k: int = 2) -> None:
        self.rag.answer(prompt, k)

    def answer_dataset(self) -> None:
        pass

    def evaluate(self) -> None:
        pass


if __name__ == "__main__":
    fire.Fire(CLI)
