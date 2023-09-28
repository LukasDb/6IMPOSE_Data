from abc import ABC, abstractmethod
from enum import Enum


class Event(Enum):
    NONE = "NONE"
    ON_SCENE_CREATED = "ON_SCENE_CREATED"
    ON_OBJECT_CREATED = "ON_OBJECT_CREATED"
    BEFORE_PHYSICS_STEP = "BEFORE_PHYSICS_STEP"
    AFTER_PHYSICS_STEP = "AFTER_PHYSICS_STEP"
    BEFORE_RENDER = "BEFORE_RENDER"
    AFTER_RENDER = "AFTER_RENDER"


class Observable:
    def __init__(self):
        self._observers: list[Observer] = []

    def notify(self, event):
        for o in [o for o in self._observers if o.event_trigger == event]:
            o.call(self)


class Observer(ABC):
    event_trigger: Event

    def __init__(self, on_event: Event) -> None:
        self.event_trigger = on_event

    def listen_to(self, caller: Observable):
        caller._observers.append(self)

    @abstractmethod
    def call(self, caller: Observable):
        pass
