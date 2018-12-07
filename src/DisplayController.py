from abc import ABC, abstractmethod


class DisplayController(ABC):
    """
    An abstract base class for all display controllers. Display controllers interface directly with frontend displays or
    other display controllers to pass animation data to the frontend
    """

    @abstractmethod
    def start(self):
        pass