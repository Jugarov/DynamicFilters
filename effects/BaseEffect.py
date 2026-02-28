from abc import ABC, abstractmethod

class BaseEffect(ABC):
    @abstractmethod
    def apply(self, frame, landmarks):
        pass