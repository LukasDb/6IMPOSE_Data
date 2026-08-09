import time
from abc import ABC, abstractmethod
from enum import Enum

import simpose as sp


class Event(Enum):
    NONE = "NONE"
    ON_SCENE_CREATED = "ON_SCENE_CREATED"
    ON_OBJECT_CREATED = "ON_OBJECT_CREATED"
    BEFORE_PHYSICS_STEP = "BEFORE_PHYSICS_STEP"
    AFTER_PHYSICS_STEP = "AFTER_PHYSICS_STEP"
    BEFORE_RENDER = "BEFORE_RENDER"
    AFTER_RENDER = "AFTER_RENDER"


class Observable:
    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def notify(self, event: Event) -> None:
        t_event_start = time.perf_counter()
        for o in [o for o in self._observers if o.event_trigger == event]:
            t0 = time.perf_counter()
            o.call(self)
            sp.logger.debug(f"{type(o).__name__}.call({event.value}) took {time.perf_counter() - t0:.2f}s")
        sp.logger.debug(f"notify({event.value}) total: {time.perf_counter() - t_event_start:.2f}s")


class Observer(ABC):
    event_trigger: Event

    def __init__(self, on_event: Event) -> None:
        self.event_trigger = on_event

    def listen_to(self, caller: Observable) -> None:
        caller._observers.append(self)

    @abstractmethod
    def call(self, caller: Observable) -> None:
        pass
