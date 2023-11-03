import click
from pathlib import Path
import subprocess
import yaml
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
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.argument("type", type=click.Choice(sp.downloaders.__datasets__))
def download(output_dir: Path, type: str):
    if type == "YCB":
        downloader = sp.downloaders.YCBDownloader(output_dir)
    else:
        raise NotImplementedError(f"Dataset {type} not implemented.")
    downloader.run()


@run.command()
@click.argument("config_file", type=click.Path(path_type=Path))
@click.option("-v", "--verbose", count=True, help="Verbosity level")
@click.option("-i", "--initialize", is_flag=True, help="Initialize a config file")
@click.option("--start_index", type=int, hidden=True)
@click.option("--end_index", type=int, hidden=True)
def generate(
    config_file: Path,
    verbose: int,
    initialize: bool,
    start_index: int | None = None,
    end_index: int | None = None,
):
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

    if config_file.exists():
        assert not initialize, f"{config_file} already exists."
        with config_file.open() as F:
            config = yaml.safe_load(F)
        generator_type = config["Generator"]["type"]

    # use the generator to initialize the config file
    else:
        assert initialize, f"{config_file} does not exist. Use -i to initialize."

        generator_type = click.prompt(
            "Generator type",
            type=click.Choice(sp.generators.__generators__),
            default=sp.generators.__generators__[0],
        )
        generator_func: type[sp.generators.Generator] = getattr(sp.generators, generator_type)
        template = generator_func.generate_template_config()
        with config_file.open("w") as F:
            F.write(template)
        return

    # TODO config overwriting
    if start_index is not None:
        config["Writer"]["params"]["start_index"] = start_index
    if end_index is not None:
        config["Writer"]["params"]["end_index"] = end_index

    generator_func: type[sp.generators.Generator] = getattr(sp.generators, generator_type)
    generator = generator_func(config=config)
    generator.start()


if __name__ == "__main__":
    run()
