"""Config round trip: every generator's scaffolded template must parse as YAML and
validate against the same pydantic models the CLI resolves at runtime."""
import pytest
import yaml

import simpose as sp


def _validate_randomizer(rand_config: dict) -> None:
    name = rand_config["type"]
    params = rand_config["params"]
    if name == "Join":
        for sub in params.values():
            _validate_randomizer(sub)
        return
    config_cls = getattr(sp.random, name + "Config")
    config_cls.model_validate(params)


@pytest.mark.parametrize("generator_name", sp.generators.__generators__)
def test_template_config_roundtrip(generator_name):
    generator_cls = getattr(sp.generators, generator_name)
    config_cls = getattr(sp.generators, generator_name + "Config")

    template = generator_cls.generate_template_config()
    config = yaml.safe_load(template)

    assert config["Generator"]["type"] == generator_name
    config_cls.model_validate(config["Generator"]["params"])

    writer_cls = getattr(sp.writers, config["Writer"]["type"])
    assert issubclass(writer_cls, sp.writers.Writer)
    sp.writers.WriterConfig.model_validate(config["Writer"]["params"])

    for rand_config in config["Randomizers"].values():
        _validate_randomizer(rand_config)


def test_checked_in_dropjects_yml_validates():
    from pathlib import Path

    yml = Path(__file__).parent.parent / "dropjects.yml"
    config = yaml.safe_load(yml.read_text())

    config_cls = getattr(sp.generators, config["Generator"]["type"] + "Config")
    config_cls.model_validate(config["Generator"]["params"])
    sp.writers.WriterConfig.model_validate(config["Writer"]["params"])
    for rand_config in config["Randomizers"].values():
        _validate_randomizer(rand_config)
