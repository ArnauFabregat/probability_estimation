import unittest
from abc import ABC, abstractmethod


class BaseTestCase(ABC, unittest.TestCase):
    @abstractmethod
    def given(self) -> None:
        raise NotImplementedError(
            "This method should be called at the beginning of the test. "
            "Here we should put code that sets up the test: Get test data, "
            "create a table or a file, etc."
        )

    @abstractmethod
    def when(self) -> None:
        raise NotImplementedError(
            "This method should be called after the 'given' method. "
            "Here we should put code that executes the piece of code we want to test."
        )

    @abstractmethod
    def then(self) -> None:
        raise NotImplementedError(
            "This method should be called after the 'when' method. "
            "Here we should put code that verifies the correct behaviour of the piece "
            "of code we are testing: Verify that some file or table was created, "
            "verify a numeric response, etc."
        )
