"""Some useful plotting functions."""

import matplotlib.pyplot as plt
from astropy import units as u
from matplotlib.figure import Figure

from .solver import Solution
from .types import FreckllArray


def plot_tp_and_kzz(
    pressure: u.Quantity,
    temperature: u.Quantity,
    kzz: u.Quantity,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot temperature and Kzz profiles.

    Args:
        pressure: Pressure profile.
        temperature: Temperature profile.
        kzz: Kzz profile.
        kzz_pressure: Pressure profile for Kzz. If None, use the same as pressure.
        ax: Matplotlib Axes object to plot on. If None, create a new one.

    Returns:
        Matplotlib Axes object with the plot.
    """
    import matplotlib.pyplot as plt
    from astropy import units as u

    if ax is None:
        fig, ax = plt.subplots()

    pressure = pressure.to(u.bar)
    # Plot temperature
    tplot = ax.plot(temperature.value, pressure.value, label="Temperature", color="red")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax2 = ax.twiny()

    # Plot Kzz
    kzzplot = ax2.plot(kzz.value, pressure.value, label="Kzz", color="blue")
    ax2.set_xlabel(f"Kzz ({kzz.unit.to_string('latex')})")
    ax2.set_xscale("log")

    lns = tplot + kzzplot
    labs = [lab.get_label() for lab in lns]
    ax.legend(lns, labs, loc=0)
    # Set labels and title

    ax.set_title("Temperature and Kzz Profiles")

    return ax


def plot_vmr(
    vmr: u.Quantity,
    pressure: u.Quantity,
    species: list[str],
    species_to_plot: list[str] = ("H2O", "CO2", "CH4", "H2", "H", "CO", "NH3", "HCN", "N2"),
    xlims: tuple[float, float] = (1e-16, 1),
    initial_vmr: u.Quantity | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot volume mixing ratio (VMR) profiles.

    Args:
        vmr: Volume mixing ratio profile.
        pressure: Pressure profile.
        species: List of species names.
        ax: Matplotlib Axes object to plot on. If None, create a new one.

    Returns:
        Matplotlib Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    pressure = pressure.to(u.bar)
    mol_index = [species.index(x) for x in species if x in species_to_plot]
    for x in mol_index:
        if initial_vmr is not None:
            ax.plot(initial_vmr[x], pressure.value, "--", alpha=0.5)
            ax.plot(vmr[x], pressure.value, label=species[x], color=ax.lines[-1].get_color())
        else:
            ax.plot(vmr[x], pressure.value, label=species[x])

    ax.set_xlabel("Volume Mixing Ratio")
    ax.set_ylabel("Pressure (bar)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(xlims)
    ax.set_title("VMR Profiles")
    ax.invert_yaxis()

    return ax


def animate_vmr(
    vmrs: u.Quantity,
    times: u.Quantity,
    pressure: u.Quantity,
    species: list[str],
    species_to_plot: list[str] = ("H2O", "CO2", "CH4", "H2", "H", "CO", "NH3", "HCN", "N2"),
    xlims: tuple[float, float] = (1e-16, 1),
    initial_vmr: u.Quantity | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Animate volume mixing ratio (VMR) profiles.

    Args:
        vmrs: Volume mixing ratio over `times`.
        times: Evaluated time points.
        pressure: Pressure profile.
        species: List of species names.
        species_to_plot: List of species to plot.
        xlims: X-axis limits.
        initial_vmr: Initial volume mixing ratio. If None, do not plot.
        ax: Matplotlib Axes object to plot on. If None, create a new one.

    Returns:
        Matplotlib Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    from matplotlib.animation import FuncAnimation

    ax.set_xlim(xlims)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("VMR")
    ax.set_ylabel("Pressure (mbar)")
    ax.set_title("Time Evolution of VMR")

    lines = []
    mol_index = [species.index(x) for x in species if x in species_to_plot]
    for x in mol_index:
        if initial_vmr is not None:
            ax.plot(initial_vmr[x], pressure.value, alpha=0.5)
            (line,) = ax.plot([], [], label=species[x], color=ax.lines[-1].get_color())
        else:
            line = ax.plot([], [], "--", alpha=0.5)

        lines.append(line)
    ax.legend()
    ax.set_title(f"Time: {times[0]:.2E} s")

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        new_fm = vmrs[frame]
        for idx, x in enumerate(mol_index):
            lines[idx].set_data(new_fm[x].T, pressure.value)
        ax.set_title(f"Time: {times[frame - 1]:.2E} s")
        return lines

    ani = FuncAnimation(fig, update, frames=len(vmrs), init_func=init, **kwargs)

    return ani


def plot_mu(
    final_vmr: FreckllArray,
    masses: u.Quantity,
    pressure: u.Quantity,
    initial_vmr: FreckllArray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the mean molecular weight.

    Args:
        final_vmr: Final volume mixing ratio.
        masses: Molecular masses of the species.
        pressure: Pressure profile.
        initial_vmr: Initial volume mixing ratio. If None, do not plot.
        ax: Matplotlib Axes object to plot on. If None, create a new one.

    Returns:
        Matplotlib Axes object with the plot.

    """
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()

    pressure = pressure.to(u.bar)
    mu_final = np.sum(final_vmr * masses[:, None], axis=0)

    if initial_vmr is not None:
        mu_initial = np.sum(initial_vmr * masses[:, None], axis=0)
        ax.plot(mu_initial, pressure.value, "--", label="Initial", alpha=0.5)

    ax.plot(mu_final, pressure.value, label="Final")

    ax.set_yscale("log")

    ax.set_xlabel(f"Mean Molecular Weight ({mu_final.unit.to_string('latex')})")
    ax.set_ylabel("Pressure (bar)")

    ax.legend()
    ax.invert_yaxis()
    ax.set_title("Mean Molecular Weight")

    return ax


def plot_stellar_flux(
    stellar_flux: u.Quantity,
    wavelength: u.Quantity,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the stellar flux."""
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(wavelength.value, stellar_flux.value)
    ax.set_xlabel(f"Wavelength ({wavelength.unit.to_string('latex')})")
    ax.set_ylabel(f"Flux ({stellar_flux.unit.to_string('latex')})")
    ax.set_yscale("log")
    ax.set_title("Incident Stellar Flux")
    return ax


def plot_solution_combined(
    solution: Solution,
    species_to_plot: list[str] = ("H2O", "CO2", "CH4", "H2", "H", "CO", "NH3", "HCN", "N2"),
    molecule_xlims: tuple[float, float] = (1e-16, 1),
    **kwargs,
) -> Figure:
    """Plot the solution."""

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        **kwargs,
    )

    axs = ax.flatten()

    # else:
    #     fig, ax = plt.subplots(nrows=1, ncols=3, **figure_args)
    #     axs = ax.flatten()

    plot_tp_and_kzz(
        solution["pressure"],
        solution["temperature"],
        solution["kzz"],
        ax=axs[0],
    )

    plot_vmr(
        solution["vmr"][-1],
        solution["pressure"],
        species=solution["species"],
        species_to_plot=species_to_plot,
        xlims=molecule_xlims,
        initial_vmr=solution["initial_vmr"],
        ax=axs[1],
    )

    plot_mu(
        solution["vmr"][-1],
        solution["masses"],
        solution["pressure"],
        initial_vmr=solution["initial_vmr"],
        ax=axs[2],
    )

    if "stellar_flux" in solution:
        plot_stellar_flux(
            solution["stellar_flux"]["incident_flux"],
            solution["stellar_flux"]["wavelength"],
            ax=axs[3],
        )

    fig.tight_layout()

    return fig
