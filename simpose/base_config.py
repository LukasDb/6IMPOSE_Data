from pydantic import BaseModel, Field
import yaml
from abc import abstractmethod


class BaseConfig(BaseModel, extra="forbid"):
    @staticmethod
    @abstractmethod
    def get_description() -> dict[str, str]:
        pass

    @classmethod
    def dump_with_comments(cls: type["BaseConfig"], **kwargs):
        self = cls(**kwargs)
        yaml_str = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, line_break="\n")

        description = self.get_description()
        annotated_lines = []

        for line in yaml_str.split("\n"):
            for key, hint in description.items():
                if key in line:
                    line += f"  # {hint}"
            annotated_lines.append(line)

        return "\n".join(annotated_lines)
