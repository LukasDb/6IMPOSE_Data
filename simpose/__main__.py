import click
from pathlib import Path
import subprocess


@click.group()
def run():
    pass


@run.command()
@click.argument("data_dir", type=click.Path(exists=True))
def view(data_dir: Path):
    print(f"viewing {data_dir}")
    import simpose.dataset_viewer as ds_viewer

    # run streamlit run dataset_viewer
    subprocess.run(
        ["streamlit", "run", ds_viewer.__file__, "--server.runOnSave", "true", "--", data_dir]
    )


if __name__ == "__main__":
    run()
