from typing import Optional, List, Generic, TypeVar
import numpy as np


T = TypeVar('T')


class Queue(Generic[T]):
    """Simple queue implementation.

    Additional features include on-line calculation of
    the average and limiting the size of the queue."""

    def __init__(self, data=Optional[List[T]], size_limit: Optional[int]=None):
        self.size_limit = size_limit
        self._data = [] if data is None else data  # type: List[T]
        self.size = 0

        if (size_limit is not None and
            data is not None and
            len(data) > size_limit):
            self._data = data[len(data) - size_limit:]

    def push(self, point: T):
        self._data.insert(0, point)
        if self.size_limit is not None and self.size >= self.size_limit:
            self.pop()
            return

        self.size += 1

    def pop(self) -> Optional[T]:
        if self.size < 0:
            return None

        self.size -= 1
        return self._data.pop()

    def clean(self):
        self._data = []
        self.size = 0

    @property
    def data(self) -> List[T]:
        return self._data[:]
