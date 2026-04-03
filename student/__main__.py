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
            print(docs)
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
        save_directory: str = "data/output",
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

    def answer_dataset(self) -> None:
        pass

    def evaluate(self) -> None:
        pass


if __name__ == "__main__":
    fire.Fire(CLI)
