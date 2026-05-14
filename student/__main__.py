import json
import fire
from student.rag import RAG
from student.validator import (
    AnsweredQuestion,
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)


class CLI:
    def __init__(self) -> None:
        """Initialize the CLI with a RAG engine instance."""
        self.rag = RAG()

    def index(
        self, path: str = "data/raw/vllm-0.10.1/", max_chunk_size: int = 2000
    ) -> None:
        """Index files in a folder and build the retrieval stores.

        Args:
            path: Root folder containing files to index.
            max_chunk_size: Maximum size for each chunk.
        """
        try:
            self.rag.index(path, max_chunk_size)
        except (FileNotFoundError, PermissionError) as e:
            print("[Error]: Error while trying to open the file/folder:", e)
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except (Exception, KeyboardInterrupt) as e:
            print("[Error]:", e)

    def search(self, query: str, k: int) -> None:
        """Search for sources matching a query and save results.

        Args:
            query: The question or search query.
            k: Number of sources to retrieve.
        """
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
        except (Exception, KeyboardInterrupt) as e:
            print("[Error]:", e)

    def search_dataset(
        self,
        dataset_path: str = (
            "data/datasets/UnansweredQuestions/dataset_docs_public.json"
        ),
        k: int = 1,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search all questions in a dataset and save retrieval results.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of sources to retrieve per question.
            save_directory: Folder where results will be written.
        """
        try:
            self.rag.search_dataset(dataset_path, k, save_directory)
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except (Exception, KeyboardInterrupt) as e:
            print("[Error]:", e)

    def answer(self, prompt: str, k: int = 2) -> None:
        """Generate an answer for a prompt and save the result.

        Args:
            prompt: The question to answer.
            k: Number of sources to retrieve.
        """
        try:
            self.rag.answer(prompt, k)
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except (Exception, KeyboardInterrupt) as e:
            print("[Error]:", e)

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/output/search_results/dataset_docs_public.json"
        ),
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a dataset of search results.

        Args:
            student_search_results_path: Path to search results JSON.
            save_directory: Folder where answers will be written.
        """
        try:
            self.rag.answer_dataset(
                student_search_results_path, save_directory
            )
        except (FileExistsError, NotADirectoryError) as e:
            print("[Error]: Error while saving index", e)
        except (Exception, KeyboardInterrupt) as e:
            print("[Error]:", e)

    def inter(self, src: MinimalSource, ret_src: MinimalSource) -> float:
        """Compute overlap ratio between a source and a retrieved source.

        Args:
            src: Ground-truth source span.
            ret_src: Retrieved source span.

        Returns:
            Overlap ratio in the range [0, 1].
        """
        index_starts = (
            src.first_character_index,
            ret_src.first_character_index,
        )
        index_ends = (
            src.last_character_index,
            ret_src.last_character_index,
        )
        src_len = index_ends[0] - index_starts[0]
        if src_len == 0:
            return 0
        return max(0.0, min(index_ends) - max(index_starts)) / src_len

    def evaluate(
        self,
        student_answer_path: str = (
            "data/output/search_results/dataset_docs_public.json"
        ),
        dataset_path: str = (
            "data/datasets/AnsweredQuestions/dataset_docs_public.json"
        ),
    ) -> None:
        """Compute and print recall for a student search result file.

        Args:
            student_answer_path: Path to student search results JSON.
            dataset_path: Path to the answered dataset JSON.
        """
        try:
            with open(student_answer_path, "r") as f:
                search_results = StudentSearchResults(**json.loads(f.read()))
            with open(dataset_path, "r") as f:
                dataset = RagDataset(**json.loads(f.read()))
            if search_results and dataset:
                total_recall = 0.0
                for i, question in enumerate(dataset.rag_questions):
                    if not isinstance(question, AnsweredQuestion):
                        continue
                    srcs_found = 0
                    srcs = question.sources
                    retrieved_srcs = search_results.search_results[
                        i
                    ].retrieved_sources

                    for src in srcs:
                        for ret_src in retrieved_srcs:
                            if src.file_path == ret_src.file_path:
                                if self.inter(src, ret_src) >= 0.05:
                                    srcs_found += 1
                                    break
                    total_recall += (
                        srcs_found / len(srcs) if len(srcs) > 0 else 0
                    )
                num_questions = len(dataset.rag_questions)
                print(
                    f"Recall@{search_results.k}:",
                    total_recall / num_questions if num_questions > 0 else 0,
                )
        except (FileNotFoundError, PermissionError) as e:
            print("[Error]: Error while loading files:", e)
        except (Exception, KeyboardInterrupt) as e:
            print("[Error]:", e)


if __name__ == "__main__":
    fire.Fire(CLI)
