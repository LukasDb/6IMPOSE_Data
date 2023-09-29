import simpose as sp
import logging
from abc import ABC, abstractmethod
from pydantic import BaseModel, validator
from enum import Enum
from simpose.observers import Observer, Event
from simpose import base_config


class RandomizerConfig(base_config.BaseConfig):
    trigger: Event


class Randomizer(Observer, ABC):
    def __init__(
        self,
        params: RandomizerConfig,
    ):
        super().__init__(on_event=params.trigger)

    @abstractmethod
    def call(self, scene: sp.Scene):
        pass


class JoinableRandomizer(Randomizer):
    @abstractmethod
    def __add__(self, other: Randomizer):
        pass
