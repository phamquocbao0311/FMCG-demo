from abc import abstractmethod
from utils.files import BaseFile


class BaseModel:

    def __init__(self, base_file: BaseFile):
        self.model = base_file.load()

    @abstractmethod
    def predict(self, item):
        raise NotImplemented
