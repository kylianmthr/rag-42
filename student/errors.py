class EmptyFolder(Exception):
    def __init__(self, message: str) -> None:
        """Create an EmptyFolder exception.

        Args:
            message: Error message describing the failure.
        """
        super().__init__(message)
