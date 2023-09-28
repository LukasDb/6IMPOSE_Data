import click
from pathlib import Path
import subprocess
import yaml
from pydantic import validate_call
import functools
import simpose as sp


@click.group()
def run():
    pass


@run.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
def view(data_dir: Path):
    print(f"viewing {data_dir}")
    import simpose.dataset_viewer as ds_viewer

    # run streamlit run dataset_viewer
    subprocess.run(
        [
            "streamlit",
            "run",
            ds_viewer.__file__,
            "--server.runOnSave",
            "true",
            "--server.headless",
            "true",
            "--",
            data_dir,
        ]
    )


@run.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", count=True)
def generate(config_file: Path, verbose: int, **kwargs):
    if verbose == 0:
        level = 30
    elif verbose == 1:
        level = 20
    elif verbose == 2:
        level = 10
    else:
        level = 1

    for handler in sp.logger.handlers:
        handler.setLevel(level)

    sp.logger.setLevel(level)

    with config_file.open() as F:
        config = yaml.safe_load(F)

    # TODO config overwriting

    # load randomizers
    randomizers = {}
    for rand_name, rand_initializer in config["Randomizers"].items():
        randomizers[rand_name] = get_randomizer(rand_initializer)

    # load writer
    writer_name = config["Writer"]["type"]
    writer_config = sp.writers.WriterParams.model_validate(config["Writer"]["params"])
    writer: sp.writers.Writer = getattr(sp.writers, writer_name)(writer_config)

    # initialize and run generator
    generator_func: type[sp.generators.Generator] = getattr(
        sp.generators, config["Generator"]["type"]
    )
    params_func: type[sp.generators.GeneratorParams] = getattr(
        sp.generators, config["Generator"]["type"] + "Config"
    )
    gen_params = params_func.model_validate(config["Generator"]["params"])
    generator = generator_func(writer=writer, randomizers=randomizers, params=gen_params)

    generator.start()


@validate_call
def get_randomizer(initializer: dict[str, str | dict]) -> sp.random.Randomizer:
    name = initializer["type"]
    config = initializer["params"]

    if name == "Join":
        assert not isinstance(config, str)
        rands = [get_randomizer(rand) for _, rand in config.items()]
        assert all(isinstance(x, sp.random.JoinableRandomizer) for x in rands)
        randomizer = functools.reduce(lambda x, y: x + y, rands)  # type: ignore

        return randomizer

    assert isinstance(config, dict)
    assert isinstance(name, str)

    class_randomizer: type[sp.random.Randomizer] = getattr(sp.random, name)
    class_config: type[sp.random.RandomizerConfig] = getattr(sp.random, name + "Config")
    cnf = class_config.model_validate(config)
    return class_randomizer(cnf)


if __name__ == "__main__":
    run()
