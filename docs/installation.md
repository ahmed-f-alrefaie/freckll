# Installation

FRECKLL has support for Python 3.9 and above. It is recommended to use a virtual environment to avoid package conflicts.

## PyPi

FRECKLL can be installed using standard `pip` command:

```bash
pip install freckll[recommended]
```

which will install FRECKLL and additional optional dependancies.

You can also be more specific and install only the required dependencies:

```bash
pip install freckll     # Minimum dependencies
pip install freckll[recommended] # ace and plot
pip install freckll[ace] # ACE chemistry
pip install freckll[plot] # Plotting
```

## Requirements

FRECKLL has the following strict requirements which will be automatically installed when using `pip`:

- Python>=3.9
- [astropy>=5.3.2](https://www.astropy.org)
- [molmass>=2024.10.25](https://pypi.org/project/molmass/)
- [numba>=0.60.0](https://numba.pydata.org)
- [numpy>=2.0.2](https://numpy.org)
- [scipy>=1.13.1](https://scipy.org)

The following optional dependencies are also available:

### ace

Installs the [ACE](https://ui.adsabs.harvard.edu/abs/2012A%26A...548A..73A/abstract) chemistry package.
This is required for the using equilibrium chemistry as the initial start point.

- [acepython>=0.0.17](https://github.com/ahmed-f-alrefaie/acepython)

### plot

Installs the plotting dependencies for plotting the results of the simulation.

- [matplotlib>=3.9.4](https://matplotlib.org/)

## From source

To install from source, clone the repository and install using `pip`:

```bash
git clone https://github.com/ahmed-f-alrefaie/freckll.git
cd freckll
pip install .[recommended]
```
