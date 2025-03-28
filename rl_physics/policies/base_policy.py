import abc

class BasePolicy(object, metaclass=abc.ABCMeta):
    def save(self, filepath: str):
        raise NotImplementedError
