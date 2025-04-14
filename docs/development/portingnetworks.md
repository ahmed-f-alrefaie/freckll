# Porting networks


## Introduction

FRECKLL has been designed to be extensible and also to allow for the addition of new networks. The philosphy of the code is that
the library contains all of the building blocks to solve the kinetic network and all the user must do is translate it into the 
form of FRECKLL.

For example, the inbuilt `VenotChemicalNetwork` is simply a convienient wrapper of the `Network` class that parses the correct reactions into the correct FRECKLL functions
and then adds them to the network.

```python
class VenotChemicalNetwork(ChemicalNetwork):
    """A chemical network from Olivia Venot."""

    def __init__(self, network_path: pathlib.Path) -> None:
        """Initialize the network.

        Args:
            network_path: The path to the network.

        """
        from .io import infer_composition, load_composition, load_efficiencies, load_nasa_coeffs, load_reactions

        network_path = pathlib.Path(network_path)
        if not network_path.is_dir():
            raise ValueError(f"{network_path} is not a directory")
        composes_file = network_path / "composes.dat"
        nasa_file = network_path / "coeff_NASA.dat"
        efficiencies = network_path / "efficacites.dat"

        if not composes_file.exists():
            print("Inferring network from directory structure.")
            composition = infer_composition(network_path)
        else:
            composition = load_composition(composes_file)

        efficiencies = load_efficiencies(efficiencies, composition)
        nasa_coeffs = load_nasa_coeffs(nasa_file)
        reactions = load_reactions(composition, network_path, efficiencies)

        super().__init__(composition, nasa_coeffs, reactions)
```

The key three things a network requires are:

- The decoded species inside the network
- The thermochemical data of each species
- The reactions


## Decoding Species

FRECKLL makes use of a `SpeciesFormula` class (a subclass of `Formula` from [molmass](https://github.com/cgohlke/molmass)) that automatically decomposes, and computes the mass and diffusion volume of the species.
For example:

```python
from freckll.species import SpeciesFormula

h2o = SpeciesFormula("H2O")
print(f"Formula: {h2o}")
print(f"Mass: {h2o.mass} g/mol")
print(f"Mono mass: {h2o.monoisotopic_mass} g/mol")
print(f"Diffusion volume: {h2o.diffusion_volume}")
print(h2o.composition_dict)
```
```
Formula: H2O
Mass: 18.015287 g/mol
Mono mass: 18.01056468403 g/mol
Diffusion volume: 13.1
{'H': (2, 2.015882, 0.11189841161009535), 'O': (1, 15.999405, 0.8881015883899046)}
```

The `formula` argument can be any valid hill-notation chemical formula. For example one of the [most important molecules in the universe](https://en.wikipedia.org/wiki/Caffeine) can be written as:
```python
from freckll.species import SpeciesFormula
mol = SpeciesFormula("C8H10N4O2")
print(f"Formula: {mol}")
print(f"Mass: {mol.mass} g/mol")
print(f"Mono mass: {mol.monoisotopic_mass} g/mol")
print(f"Diffusion volume: {mol.diffusion_volume}")
print(mol.composition_dict)
```
```
Formula: C8H10N4O2
Mass: 194.19095199999998 g/mol
Mono mass: 194.08037557916 g/mol
Diffusion volume: 180.68
{'C': (8, 96.08592, 0.49480122019279255), 'H': (10, 10.07941, 0.05190463250831584), 'N': (4, 56.026812, 0.2885140189229826), 'O': (2, 31.99881, 0.1647801283759091)}
```

The class has some special properties. For example we can use a string version to compare.
```python
from freckll.species import SpeciesFormula

h2o = SpeciesFormula("H2O")

print(h2o == "H2O") # True
```


For 99% of the time, the default argument is enough. However there will be cases where different isotopologues or electronic state must be represented. For example in
the Venot network we have species:

- $O(^3P)$
- $O(^1D)$

Which are presented as `O3P` and `O1D` in the network. The `SpeciesFormula` class by default may interpret this as `O3` and `P` or `O1` and `D`. 
For exmaple:
```python
from freckll.species import SpeciesFormula
o3p = SpeciesFormula("O3P")

print(o3p.composition_dict)
```
```
{'O': (3, 47.998215, 0.6077879372478617), 'P': (1, 30.973761998, 0.3922120627521383)}
```

To avoid this, the user can specify the `true_formula` argument. This is a string that will be used to decode the species. For example:
```python
from freckll.species import SpeciesFormula
o3p = SpeciesFormula("O3P", true_formula="O")
o1d = SpeciesFormula("O1D", true_formula="O")
print(o3p.composition_dict)
print (o1d.composition_dict)
```
```
{'O': (1, 15.999405, 1.0)}
{'O': (1, 15.999405, 1.0)}
```

### Recommended pattern

When loading in a species, we recommend the following pattern:

- A map of exceptions
- A `_decode_species` function that takes a string and returns a `SpeciesFormula` object.

For example, with the previous example, we have this pattern:
```python
from freckll.species import SpeciesFormula

_species_exceptions = {
    "O3P": SpeciesFormula("O3P", true_formula="O"),
    "O1D": SpeciesFormula("O1D", true_formula="O"),
}

def _decode_species(species: str) -> SpeciesFormula:
    """Decode the species.

    Args:
        species: The species to decode.

    Returns:
        The decoded species.

    """
    if species in _species_exceptions:
        return _species_exceptions[species]
    else:
        return SpeciesFormula(species)  
```


## Thermochemical data

FRECKLL uses the 7-coeff [NASA polynomial](http://combustion.berkeley.edu/gri_mech/data/nasa_plnm.html) to represent the thermochemical data of a particular species.
The `freckll.NasaCoeff` class is a simple dataclass that allows you to directly assign data to the class. For example:
```python
from freckll.nasa import NasaCoeff

nasa = NasaCoeff(
    species=SpeciesFormula("H2O"),
    x1=1000.0,
    x2=2000.0,
    x3=3000.0,
    a_coeff=np.array([1,2,3,4,5,6,7])
    b_coeff=np.array([1,2,3,4,5,6,7])
)
```
You can then call the `nasa` object to get the thermochemical data $\frac{H}{RT}$ and $\frac{S}{R}$. For example:
```python
temperature = np.array([1000, 2000, 3000]) * u.K
h, s = nasa(temperature)
```

A FRECKLL `Network` requires the user to supply a list of `NasaCoeff` objects corresponding to the species in the network.
```python

species_nasa = [
    NasaCoeff(
        species=SpeciesFormula("H2O"),
        x1=1000.0,
        x2=2000.0,
        x3=3000.0,
        a_coeff=np.array([1,2,3,4,5,6,7])
        b_coeff=np.array([1,2,3,4,5,6,7])
    ),
    NasaCoeff(
        species=SpeciesFormula("CO"),
        x1=1000.0,
        x2=2000.0,
        x3=3000.0,
        a_coeff=np.array([1,2,3,4,5,6,7])
        b_coeff=np.array([1,2,3,4,5,6,7])
    )
]
...
```

### Recommended pattern

When loading in NASA coefficients, we recommend the following pattern:
- Make use of the `decode_species` function to decode the species

For example if we have nasa coeffs of this format:

```
# My fabulous NASA coefficient format
H2O 1000.0 2000.0 3000.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
CO 1000.0 2000.0 3000.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
...
```
We could load them in like this:

```python
from freckll.nasa import NasaCoeff
from typing import Callable
from freckll.species import SpeciesFormula

def load_nasa(nasa_file: str, 
    decode_species: Callable[[str], SpeciesFormula]) -> list[NasaCoeff]:
    """Load the NASA coefficients from a file.

    Args:
        nasa_file: The path to the NASA coefficients file.
        decode_species: The function to decode the species.

    Returns:
        A list of NasaCoeff objects.

    """
    with open(nasa_file, "r") as f:
        lines = f.readlines()

    nasa_coeffs = []
    for line in lines:
        if line.startswith("#"):
            continue
        data = line.split()
        species = decode_species(data[0])
        x1 = float(data[1])
        x2 = float(data[2])
        x3 = float(data[3])
        a_coeff = np.array([float(x) for x in data[4:11]])
        b_coeff = np.array([float(x) for x in data[11:18]])
        nasa_coeffs.append(NasaCoeff(species, x1, x2, x3, a_coeff, b_coeff))

    return nasa_coeffs
```

## Reactions

Reactions are represented by two objects. A `ReactionCall` which executes the reaction and a `Reaction` which contains the computed reaction data.
All reactions under `freckll.reaction` do not produce a result but instead return a function of the form:
```python

def reaction(concentration) -> list[Reaction]:
    ...
```

The usage pattern is therefore:
```python

react_func = k0kinf(<argshere>)
reaction_rate = react_func(concentration)

print(reaction_rate[0])
print(reaction_rate[1]) # If reversible

```

This is what is actually happening when reactions are *compiled*. The appropriate function is created and then is reused. Every reaction has these parameters at the end:

- `temperature`
- `pressure`
- `thermo_reactants`
- `thermo_products`

Where `thermo_reactants` and `thermo_products` are arrays of $H$ and $S$ for the reactants and products respectively. These are used to compute the reversible reaction rate.
Using this we can determine the `fixed` parameters of the reaction, the `compiled` parameters and the `variable` parameters.

Fixed:

- Reaction specific
- Do not change with regards to temperature, pressure or concentration
- Example: Arrhenius parameters

Compiled:

- Reaction specific
- Change with regards to temperature, pressure or thermochemical data
- Does not change during a solve
- Example: Reversible reactions, temperature/pressure dependance.

Variable:

- Reaction specific
- Changes every time step
- Concentration/Number density of the species

We can use partial to create a function that stores all of the fixed parameters (such as Arrhenius parameters) and then returns a function that takes
the compiled parameters (such as temperature, pressure and thermochemical data) which then returns a function that takes the variable parameters (such as concentration).
For example, we could compile a reversible many-body reaction like this:

```python
from freckll.reaction import ReactionCall
from freckll.reaction import corps_reaction
from functools import partial

species = [SpeciesFormula("O2"), SpeciesFormula("H2"), SpeciesFormula("H2O2")]

# Compiling H2 + O2 <=> H2O2

inverted = True


k_coeffs = [
    10.0, # A
    3.0, # n
    100.0, # Er
]   

reactants = [
    SpeciesFormula("H2"),
    SpeciesFormula("O2"),
]

products = [
    SpeciesFormula("H2O2"),
]

reaction_call = ReactionCall(
    species,  # All species in network
    reactants,  # Reactants
    products,  # Products
    ["corps", "reaction", "many body"], # Tags to identify reaction
    inverted, # Reaction is reversible
    partial(react.corps_reaction, k_coeffs, inverted, reactants, products), # Our partial function
)
```

This reaction call can be passed into a network to be used to compute reaction rates.

### Available reactions

All reactions present are reversible if used with `invert=True`

- `k0kinf_reaction`: $k_0$ and $k_\infty$ 2-body reaction.
- `decomposition_k0kinf_reaction`: Decomposition reaction with $k_0$ and $k_\infty$.
- `corps_reaction`: Many body reaction
- `k0_reaction`: 2-body reaction with $k_0$.
- `decomposition_reaction`: Decomposition reaction with $k_0$.
- `de_excitation_reaction`: De-excitation reaction with $k_0$.
- `manybody_plog_reaction`: Many body reaction with PLOG.
- `decomposition_plog_reaction`: Decomposition reaction with PLOG.

Additionally reactions with $k_0$ and $k_\infty$ can use one of two falloff functions from `freckl.reactions.falloff`:

- `troe_falloff_term`: Troe falloff term
- `sri_falloff_term`: Stanford Research Institute falloff function



### Recommended pattern

When loading in reactions, we recommend the following pattern:
- Using the `decode_species` function to decode the species
- A `map_reaction` that maps a reaction to the correct reaction function
- Use partial to fix the reaction parameters

For example, if we have a reaction file of the form:
```

# k_0 k_inf 
H20 + O2 <=> H2O2,  8.306E-12,  0.000E+00,  0.000E+00,  2.000E+00,  1.000E+02, 8.306E-12,  0.000E+00,  0.000E+00,  2.000E+00,  1.000E+02,
...

#Decomposition
CO2 => CO + O2,    1.000E+13,  0.000E+00,  2.363E+04,  1.260E+00,  0.000E+00
...


```

We could load them in like this:
```python

from freckll.reaction import ReactionCall
from freckll.reaction import k0kinf_reaction, decomposition_reaction
from functools import partial
from typing import Callable
from freckll.species import SpeciesFormula


def load_reactions(
    species: list[SpeciesFormula], # Species in whole network
    reaction_file: str,
    decode_species: Callable[[str], SpeciesFormula],
) -> list[ReactionCall]:

    _map_reaction = {
        "k0 kinf": k0kinf_reaction,
        "Decomposition": decomposition_reaction,
    }

    chosen_reaction = None

    reaction_calls = []

    with open(reaction_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                chosen_reaction = _map_reaction[line[1:].strip()]
                continue
            data = line.split(",")

            invert = '<=>' in line

            coeffs = [float(x) for x in data[1:]]
            reactants = data[0].split("<=>")[0].split("+")
            products = data[0].split("<=>")[1].split("+")

            reactants = [decode_species(x.strip()) for x in reactants]
            products = [decode_species(x.strip()) for x in products]

            reaction_call = ReactionCall(
                species,
                reactants,
                products,
                ["reaction"],
                invert,
                partial(chosen_reaction, coeffs, invert, reactants, products),
            )

            reaction_calls.append(reaction_call)
    return reaction_calls

```

## Creating the network

Creating a network is simply putting all of these together and passing them into the `Network` class. For example:
```python
from freckll.network import Network

loaded_species = load_species("species.txt", decode_species)
loaded_nasa = load_nasa("nasa.txt", decode_species)
loaded_reactions = load_reactions("reactions.txt", decode_species)

network = Network(
    loaded_species,
    loaded_nasa,
    loaded_reactions,
)
```
This will create a network that can be used to solve the kinetic equations. You can simply pass this into a solver
```python
from freckll.solver import RosenbrockSolver

solver = RosenbrockSolver(network)
solver.solve(
    initial_conditions,
    t_span=[0,1e10],
    ...
)
```


## Photochemistry

Photochemistry is a bit different. Photochemistry only requires two data sources: The actinic flux and the cross sections.

### Actinic flux

This is handled by the ``StarSpectra`` class, it only requires three arguments. The spectral grid and the actinic flux and the reference distance.

```python

from freckll.reactions.photo import StarSpectra
from astropy import units as u

ss = StarSpectra(
    spectral_grid=np.linspace(100, 2000, 1000) * u.nm,
    actinic_flux=np.ones(1000) * u.photon / (u.cm**2 * u.s * u.nm),
    reference_distance=1.0 * u.AU,
)
```
The spectral grid is the wavelength grid of the actinic flux and the reference distance is the distance of the measured flux. The actinic flux must be in units that
are analagous/convertable to $\frac{photons}{cm^2 s nm}$. FRECKLL will handle the conversion for you so you can use any units you like. The `StarSpectra` class will also handle the conversion of the spectral grid to the correct units.

### PhotoMolecule

`PhotoMolecule` is a class that represents a molecule with cross-section and quantum yields. 

[To be continued]









        









