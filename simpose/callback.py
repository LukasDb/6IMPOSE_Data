from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List


class CallbackType(Enum):
    NONE = auto()
    ON_SCENE_CREATED = auto()
    ON_OBJECT_CREATED = auto()
    ON_PHYSICS_STEP = auto()
    BEFORE_RENDER = auto()
    AFTER_RENDER = auto()


class Callback(ABC):
    def __init__(self, caller: "Callbacks", type: CallbackType) -> None:
        super().__init__()
        caller.add(self)
        self._type = type

    @abstractmethod
    def callback(self):
        pass


class Callbacks:
    def __init__(self) -> None:
        self._callbacks: List[Callback] = []

    def add(self, callback: Callback):
        self._callbacks.append(callback)

    def callback(self, callback_type: CallbackType):
        for cb in [x for x in self._callbacks if x._type == callback_type]:
            cb.callback()
