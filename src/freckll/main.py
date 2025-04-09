import click
from freckll import __version__
import datetime
from .log import setup_log
import pathlib
import logging
import time
_log = logging.getLogger("freckll")


@click.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input file to run.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output file to write.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite the output file if it exists.",
)
@click.option(
    "--plot",
    is_flag=True,
    default=False,
    help="Plot the results.",
)
@click.option(
    "--plot-path",
    type=click.Path(),
    default=pathlib.Path.cwd(),
    help="Path to save the plots.",
)
@click.option(
    "--plot-prefix",
    type=str,
    default="freckll",
    help="Prefix for the plot filenames.",
)
@click.option(
    "--animate",
    is_flag=True,
    default=False,
    help="Animate the results.",
)

def freckll_cli(input: pathlib.Path | str,
                output: pathlib.Path | str,
                overwrite: bool = False,
                plot: bool = False,
                plot_path: pathlib.Path | str = None,

                plot_prefix: str = None,
                animate: bool = False,
                ) -> None:
    """Run the Freckll simulation.
    
    Args:
        input: Path to the input file.
        output: Path to the output file.
        overwrite: Overwrite the output file if it exists.
        plot: Plot the results.
        plot_path: Path to save the plots.
        plot_prefix: Prefix for the plot filenames.
        animate: Animate the results.
    
    """
    import logging
    from .io.dispatcher import load_and_run_input
    from .io.output import write_solution_h5py


    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    started_at = datetime.datetime.now()
    start_time = time.time()
    _log.info(f"FRECKLL version {__version__}.")
    _log.info("Start Date: %s", started_at.isoformat())
    
    _log.info("Input file: %s", input)
    _log.info("Output file: %s", output)

    if not overwrite and pathlib.Path(output).exists():
        _log.error("Output file already exists. Use --overwrite to overwrite.")
        return


    _log.info("Beginning Solve...")

    result = load_and_run_input(input)


    if result["success"]:
        _log.info("Solve complete.")
    else:
        _log.error("Solve failed.")
        output = pathlib.Path(output)
        output = output.with_suffix(".failed.h5")
        
    _log.info("Writing output to %s", output)
    write_solution_h5py(result, output, overwrite=overwrite)
    _log.info("Output complete.")
    _log.info("Solve time: %.2f seconds", time.time() - start_time)

    if plot:
        from .plot import plot_solution_combined
        plot_path = pathlib.Path(plot_path)
        plot_prefix = f"{plot_prefix}_"
        plot_suffix = "" if result["success"] else "_failed"
        fig = plot_solution_combined(result, figsize=(10, 10))
        fig.savefig(plot_path / f"{plot_prefix}solution{plot_suffix}.png", dpi=300)
        _log.info("Plotting complete.")

    if animate:
        from .plot import animate_vmr
        plot_path = pathlib.Path(plot_path)
        plot_prefix = f"{plot_prefix}_"
        plot_suffix = "" if result["success"] else "_failed"
        animate_path = plot_path / f"{plot_prefix}animation{plot_suffix}.mp4"
        ani = animate_vmr(result["vmr"],result["times"],result["pressure"],result["species"],initial_vmr=result["initial_vmr"])
        ani.save(animate_path, fps=30, dpi=300)
        _log.info("Animation complete.")


    _log.info("End Date: %s", datetime.datetime.now().isoformat())



if __name__ == "__main__":
    freckll_cli()




