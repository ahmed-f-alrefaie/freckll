"""Molecule functions and classes."""

import enum
import typing as t

from molmass import Formula

_VT = t.TypeVar("_VT")
_T = t.TypeVar("_T")


class SpeciesState(str, enum.Enum):
    """Species state."""

    GAS = "gas"
    LIQUID = "liquid"
    SOLID = "solid"


class SpeciesFormula(Formula):
    """Represents a particular species."""

    def __init__(
        self,
        formula: str,
        state: t.Optional[SpeciesState] = None,
        true_formula: t.Optional[str] = None,
    ) -> None:
        """Initialize species.

        Args:
            formula: The formula of the species.
            state: The state of the species.
            isomer_id: The isomer id of the species.
        """
  
        
        if state is None:
            # Try to guess the state from [s] or [l] in the formula
            if formula.endswith("[s]"):
                state = SpeciesState.SOLID
                formula = formula[:-3]
            elif formula.endswith("[l]") or formula.endswith("[c]"):
                state = SpeciesState.LIQUID
                formula = formula[:-3]
            else:
                state = SpeciesState.GAS
        self.input_formula = formula
        self.true_formula = true_formula or formula
        self.state = state
        super().__init__(self.true_formula)

        self.composition_dict = self.composition().asdict()
        self.composition_values = list(self.composition_dict.values())
        self.diffusion = None
    def _compute_diffusion(self) -> None:
        """Compute the diffusion coefficient."""
        from .diffusion import diffusion_volume

        self.diffusion = diffusion_volume(self)

    @property
    def diffusion_volume(self) -> float:
        """Compute the diffusion volume."""
        if self.diffusion is None:
            self._compute_diffusion()
        return self.diffusion

    def __hash__(self) -> int:
        """Hash function. Necessary for sets and dicts."""

        return hash(self.input_formula)

    def __eq__(self, other: t.Union[str, "SpeciesFormula", object]) -> bool:
        """Equality check. Necessary for sets and dicts."""
        if isinstance(other, str):
            # Support for checking against a string and the original formula
            return (self.input_formula == other) 
        if not isinstance(other, SpeciesFormula):
            raise TypeError
        comp_self = self.composition_values
        comp_other = other.composition_values

        return (
            frozenset(comp_self) == frozenset(comp_other)
            and self.state == other.state
            and self.formula == other.formula
            and self.input_formula == other.input_formula
        )

    def __str__(self) -> str:
        str_val = f"{self.input_formula}"
        if self.state != SpeciesState.GAS:
            str_val += f"[{self.state}]"
        return str_val

    def __repr__(self) -> str:
        return f"SpeciesFormula({self.__str__()})"


def species_check(a: t.List[SpeciesFormula], b: t.List[SpeciesFormula]) -> bool:
    """Check for same species."""
    return frozenset(a) == frozenset(b)


def select_same(
    a: t.List[SpeciesFormula], b: t.List[SpeciesFormula]
) -> t.List[SpeciesFormula]:
    """Select the same species from two lists."""
    return list(frozenset(a) & frozenset(b))


class SpeciesDict(dict[SpeciesFormula, _VT]):
    """A dictionary to hold species formula.

    A standard dictionary with the ability to use a string as a key
    as well.

    """

    def get(
        self, key: t.Union[str, SpeciesFormula], default: t.Optional[_VT | _T] = None
    ) -> _VT | _T:
        if isinstance(key, str):

            if "/" in key:
                molecule
                key = SpeciesFormula(molecule.strip(), isomer_id=varient.strip())
            else:
                key = SpeciesFormula(key)
        return super().get(key, default)

    # @overload
    # def get(self, key: t.Union[str, SpeciesFormula], default: _T, /) -> _VT | _T: ...

    def __getitem__(self, key: t.Union[str, SpeciesFormula]) -> _VT:
        if isinstance(key, str):

            if "/" in key:
                molecule, varient = key.split("/")
                key = SpeciesFormula(molecule.strip(), isomer_id=varient.strip())
            else:
                key = SpeciesFormula(key)
        return super().__getitem__(key)

    def __setitem__(self, key: t.Union[str, SpeciesFormula], value: _VT) -> None:
        if isinstance(key, str):

            if "/" in key:
                molecule, varient = key.split("/")
                key = SpeciesFormula(molecule.strip(), isomer_id=varient.strip())
            else:
                key = SpeciesFormula(key)
        super().__setitem__(key, value)

    def __contains__(self, key: t.Union[str, SpeciesFormula, object]) -> bool:
        if isinstance(key, str):

            if "/" in key:
                molecule, varient = key.split("/")
                key = SpeciesFormula(molecule.strip(), isomer_id=varient.strip())
            else:
                key = SpeciesFormula(key)

        if not isinstance(key, SpeciesFormula):
            raise TypeError
        return super().__contains__(key)
