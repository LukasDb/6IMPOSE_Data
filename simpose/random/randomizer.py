from enum import Enum
import simpose as sp
from pathlib import Path
from abc import ABC, abstractmethod
from simpose.observers import Observer, Event
from simpose import base_config
import sys


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


def register_operator(cls_params: type[RandomizerConfig]):
    py_path = sys.executable
    is_blender_builtin = "Blender.app" in py_path
    if not is_blender_builtin:
        return lambda cls: cls

    def wrapper(cls: type[Randomizer]):
        import bpy

        class Operator(bpy.types.Operator):
            bl_idname = f"simpose.{cls.__name__.lower()}"
            bl_label = cls.__name__
            bl_options = {"REGISTER", "UNDO"}  # Enable undo for the operator.
            bl_parent_id = "6IMPOSE_panel"

            def execute(self, context):
                scene = context.scene

                p = {k: getattr(self, k, None) for k in cls_params.model_fields.keys()}
                p["trigger"] = sp.Event.NONE
                p = cls_params(**p)
                randomizer: Randomizer = cls(p)
                randomizer.call(sp.Scene(scene))
                return {"FINISHED"}

        # add properties according to params
        params = cls_params(trigger=sp.Event.NONE)
        for key, value in params.model_dump().items():
            if isinstance(value, int):
                prop = bpy.props.IntProperty(key, default=value)
            elif isinstance(value, float):
                prop = bpy.props.FloatProperty(key, default=value)
            elif isinstance(value, list | tuple):
                if isinstance(value[0], float):
                    prop = bpy.props.FloatVectorProperty(key, default=value, size=len(value))
                elif isinstance(value[0], int):
                    prop = bpy.props.IntVectorProperty(key, default=value, size=len(value))
            elif isinstance(value, str | Path):
                prop = bpy.props.StringProperty(key, default=str(value))
            elif isinstance(value, Enum):
                prop = bpy.props.EnumProperty(
                    items=[(e.name, e.name, e.value) for e in value.__class__],
                    name=key,
                    default=value.value,
                )
            else:
                continue
                # raise NotImplementedError(f"Type {type(value)} not implemented")
            Operator.__annotations__[key] = prop
        sp.BL_OPS.append(Operator)

        return cls

    return wrapper
