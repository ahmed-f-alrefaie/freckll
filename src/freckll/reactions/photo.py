"""Module related to loading in photodisocciation data."""
from astropy import units as u
from ..species import SpeciesFormula, SpeciesDict
import numpy as np
import numpy.typing as npt
from ..types import FreckllArray
import typing as t
from ..reactions.data import Reaction
class StarSpectra:

    def __init__(self, 
                 wavelength: u.Quantity,
                 flux: u.Quantity,
                 reference_distance: u.Quantity
                 ):
        """Initialize the star spectrum with wavelength and flux data.

        Args:
            wavelength: The wavelength of the star spectrum.
            flux: The flux of the star spectrum.
            reference_distance: The reference distance for the flux.
        
        """
        self.wavelength = wavelength
        self.flux = flux

        self.reference_distance = reference_distance

    def incident_flux(self, 
                 distance: u.Quantity) -> u.Quantity:
        """Calculate the flux density at the reference distance.

        Returns:
            u.Quantity: The flux density at the reference distance.
        
        """
        return self.flux * (self.reference_distance / distance)**2 / 2




class CrossSection:
    """Loads cross-section data for molecule."""

    def __init__(self, 
                 molecule: SpeciesFormula,
                 wavelength: u.Quantity, cross_section: u.Quantity):
        self.wavelength = wavelength
        self.cross_section = cross_section
        self.molecule = molecule
        """Initialize and load the cross-section data.

        Args:
            molecule: The molecule for which the cross-section is loaded.
            wavelength: The wavelength at which the cross-section is measured.
            cross_section: The cross-section value at the given wavelength.
        
        """

    def interp_to(self, 
                 wavelength: u.Quantity) -> "CrossSection":
        """Interpolate the cross section to the given wavelength.
        
        Args:
            wavelength: The wavelength to which the cross section is interpolated.

        Returns:
            CrossSection: A new CrossSection object with the interpolated values.
        
        
        """

        new_wavelength = self.wavelength.to(wavelength.unit, equivalencies=u.spectral())

        return CrossSection(
            self.molecule,
            wavelength,
            np.interp(wavelength.value, new_wavelength.value, self.cross_section.value, left=0.0, right=0.0) << self.cross_section.unit)

    def __add__(self, other: "CrossSection") -> "CrossSection":
        """Add two cross sections together.

        Args:
            other: The other cross section to add.

        Returns:
            CrossSection: A new CrossSection object with the summed values.
        
        """
        if not isinstance(other, CrossSection):
            raise TypeError("Can only add CrossSection objects")

        if self.molecule != other.molecule:
            raise ValueError("Cannot add cross sections of different molecules")

        other = other.interp_to(self.wavelength)

        return CrossSection(
            self.molecule,
            self.wavelength,
            self.cross_section + other.cross_section
        )


class QuantumYield:
    """Loads quantum yield data for molecule.
    
    This represents the branching ratio of the photodissociation process.

    This is generic and it is up to the chemical network to determine how branching ratios
    are organised
    
    """

    def __init__(self, 
                 molecule: SpeciesFormula, 
                 branch_id: int | str,
                 wavelength: u.Quantity, qy: npt.NDArray[np.float64]):
        """Initialize and load the quantum yield data.

        Args:
            molecule: The molecule for which the quantum yield is loaded.
            branch_id: The ID of the branching ratio.
            wavelength: The wavelength at which the quantum yield is measured.
            qy: The quantum yield value at the given wavelength.
        """
        self.wavelength = wavelength
        self.qy = qy
        self.molecule = molecule
        self.branch_id = branch_id

    def interp_to(self,
                    wavelength: u.Quantity) -> "QuantumYield":
            """Interpolate the quantum yield to the given wavelength.
            
            Args:
                wavelength: The wavelength to which the quantum yield is interpolated.

            Returns:
                QuantumYield: A new QuantumYield object with the interpolated values.
            
            """
    
            new_wavelength = self.wavelength.to(wavelength.unit, equivalencies=u.spectral())
    
            return QuantumYield(
                self.molecule,
                self.branch_id,
                wavelength,
                np.interp(wavelength.value, new_wavelength.value, self.qy, left=0.0, right=0.0) )



class PhotoMolecule:
    """Represents a molecule used in photodissociation reactions.

    This class contains the cross-section data and quantum yields for the molecule.

    """
    def __init__(self, 
                molecule: SpeciesFormula,
                cross_section: CrossSection):
        """Initialize the PhotoMolecule with a molecule and its cross-section.
        
        Args:
            molecule: The molecule for which the cross-section is loaded.
            cross_section: The cross-section data for the molecule.
            
        
        
        """
        self.molecule = molecule
        self.cross_section = cross_section
        self.quantum_yields: dict[str | int, QuantumYield]  = {}

    def add_quantum_yield(self,
                          branch_id: int | str,
                          quantum_yield: QuantumYield):
        """Add a quantum yield to the molecule.
        
        Will interpolate the quantum yield to the cross-section wavelength.

        Args:
            branch_id: The ID of the branching ratio.
            quantum_yield: The quantum yield data to be added.
        
        
        """
        self.quantum_yields[branch_id] = quantum_yield.interp_to(self.cross_section.wavelength)
    def get_quantum_yield(self, branch_id: int | str) -> QuantumYield:
        """Get the quantum yield for a given branch.
        
        If the quantum yield is not found, a default quantum yield of 1.0 is returned.

        Args:
            branch_id: The ID of the branching ratio.
        Returns:
            QuantumYield: The quantum yield data for the given branch.
        
        
        """

        if branch_id not in self.quantum_yields:
             return QuantumYield(self.molecule, branch_id, self.cross_section.wavelength, np.ones(self.cross_section.wavelength.shape))

        return self.quantum_yields[branch_id]
    
    def interp_to(self, wavelength: u.Quantity) -> "PhotoMolecule":
        """Interpolate the cross section and quantum yields to the given wavelength.
        
        Args:
            wavelength: The wavelength to which the cross section and quantum yields are interpolated.

        Returns:
            PhotoMolecule: A new PhotoMolecule object with the interpolated values.
        
        
        """

        new_cross_section = self.cross_section.interp_to(wavelength)
        new_molecule = PhotoMolecule(self.molecule, new_cross_section)

        for branch_id, qy in self.quantum_yields.items():
            new_molecule.add_quantum_yield(branch_id, qy.interp_to(wavelength))

        return new_molecule



    def reaction_rate(self, branch_id: int | str, flux: u.Quantity) -> list[FreckllArray]:
        r"""Compute the reaction rate for a given branch.

        Calculates the integral:

        $$
        R = \int \sigma(\lambda) \cdot QY(\lambda) \cdot F(\lambda) d\lambda
        $$

        where:
        - $R$ is the reaction rate
        - $\sigma(\lambda)$ is the cross-section
        - $QY(\lambda)$ is the quantum yield
        - $F(\lambda)$ is the flux
        The integral is computed over the wavelength range of the cross-section.

        Args:
            branch_id: The ID of the branching ratio.
            flux: The flux at which the reaction rate is computed.

        Returns:
            u.Quantity: The reaction rate for the given branch.
        
        
        """
        qy = self.get_quantum_yield(branch_id)
        flux = flux.to(u.photon / u.cm**2 / u.s / u.nm, equivalencies=u.spectral_density(self.cross_section.wavelength))
        reaction_rate = np.trapezoid(qy.qy * self.cross_section.cross_section * flux,self.cross_section.wavelength, axis=-1)/u.photon
        return reaction_rate.to(1/(u.s)).value


class PhotoReactionCall:

    def __init__(self, 
                 reactant: PhotoMolecule,
                 products: list[SpeciesFormula],
                 branch_id: int | str,
                 species_list: list[SpeciesFormula] = None,
                 reactant_index: int = None,
                 product_indices: npt.NDArray[np.integer] = None,
                 tags: list[str] = None):
        """Initialize the photodissociation reaction call.

        Args:
            reactants: The reactants of the reaction.
            products: The products of the reaction.
            branch_id: The ID of the branching ratio.
            species_list: The list of species in the network.
            reactant_index: The index of the reactant in the species list.
            product_indices: The indices of the products in the species list.
            tags: The tags associated with the reaction.
        
        """
        self.reactant = reactant
        self.products = products
        self.tags = tags if tags is not None else ["photodissociation"]
        self.reactant_index = reactant_index or species_list.index(reactant.molecule)
        self.product_indices = product_indices or np.array([species_list.index(x) for x in products])
        self.branch_id = branch_id
        self.tags = list(set(self.tags))

    def interpolate_to(self, wavelength: u.Quantity) -> "PhotoReactionCall":
        """Interpolate the reaction call to the given wavelength.
        
        Args:
            wavelength: The wavelength to which the reaction call is interpolated.

        Returns:
            PhotoReactionCall: A new PhotoReactionCall object with the interpolated values.
        
        
        """
        reactant = self.reactant.interp_to(wavelength)

        return PhotoReactionCall(
            reactant,
            self.products,
            self.branch_id,
            reactant_index=self.reactant_index,
            product_indices=self.product_indices,
            tags=self.tags,
        )


    def __call__(self, flux: u.Quantity, number_density: FreckllArray) -> Reaction:
        """Call the reaction.

        Args:
            concentration: The concentration of the reactants.

        Returns:
            FreckllArray: The reaction rate.
        
        """
        reaction_rate = self.reactant.reaction_rate(self.branch_id, flux)

        reaction = Reaction(
            reactants=[self.reactant.molecule],
            products=self.products,
            reactants_indices=np.array([self.reactant_index]),
            product_indices=self.product_indices,
            reaction_rate=reaction_rate,
            tags=self.tags,
        )

        reaction.calculate_density_krate(number_density)

        return [reaction]




def rayleigh(
        spectral_grid: u.Quantity,
        alpha: u.Quantity, depolar_factor: float) -> u.Quantity:
    r"""Calculate the Rayleigh scattering cross-section.
    
    The Rayleigh scattering cross-section is given by the formula:
    $$
    \sigma_R = \frac{8 \pi^3}{3} \left(\frac{\alpha}{\lambda}\right)^4 (1 + \delta^2)$$
    where:
    - $\sigma_R$ is the Rayleigh scattering cross-section
    - $\alpha$ is the polarizability of the molecule    
    
    Args:
        spectral_grid: The wavelength grid.
        alpha: The polarizability of the molecule.
        depolar_factor: The depolarization factor.
    Returns:
        u.Quantity: The Rayleigh scattering cross-section.
    
    """

    wavelength_cm = spectral_grid.to(u.cm, equivalencies=u.spectral())

    alpha = alpha.to(u.cm**3)

    return 8*np.pi/3 * (2*np.pi/wavelength_cm)**4 * alpha**2 * depolar_factor


rayleigh_species = SpeciesDict[t.Callable[
    [CrossSection],u.Quantity]]()
rayleigh_species.update(
    {
        SpeciesFormula("N2"): lambda x: CrossSection(SpeciesFormula("N2"), x, rayleigh(x, 1.76e-24 * u.cm**3, 1.0518)),
        SpeciesFormula("He"): lambda x: CrossSection(SpeciesFormula("He"), x, rayleigh(x, 0.21e-24 * u.cm**3, 1.0)),
        SpeciesFormula("H2"): lambda x: CrossSection(SpeciesFormula("H2"), x, rayleigh(x, 0.82e-24 * u.cm**3, 1.0341)),
    }

)


def optical_depth(
    altitude: u.Quantity,
    number_density: u.Quantity,
    cross_sections: u.Quantity,
    cross_section_indices: npt.NDArray[np.integer],
) -> FreckllArray:
    dz = np.zeros_like(altitude)
    dz[:-1] = np.diff(altitude)
    dz[-1] = dz[-2]

    number_density_dz = number_density * dz

    cross_section_density = cross_sections * number_density_dz[cross_section_indices]

    tau = np.sum(cross_section_density, axis=0)

    return tau
    


def radiative_transfer(
    flux_top: u.Quantity,
    optical_depth: FreckllArray,
    incident_angle: u.Quantity,
    albedo: float | npt.NDArray[np.float64] = 0.0,
) -> u.Quantity:
    r"""Computes the radiative transfer equation.
    
    Computes the radiative transfer equation for a given flux and optical depth.

    The equation is given by:
    $$
    F_{down} = F_{top} \cdot e^{-\tau}$$
    where:
    - $F_{down}$ is the flux at the bottom of the atmosphere
    - $F_{top}$ is the flux at the top of the atmosphere
    - $\tau$ is the optical depth
    The flux is then propagated upwards and downwards through the atmosphere.
    The albedo is taken into account for the upward flux.
    Args:
        flux_top: The flux at the top of the atmosphere.
        optical_depth: The optical depth of the atmosphere.
        incident_angle: The angle of incidence.
        albedo: The albedo of the surface.
    Returns:
        u.Quantity: The total flux at each layer.
    
    """

    tau = np.exp(-optical_depth) / np.cos(incident_angle)

    flux_down = np.zeros_like(flux_top)
    flux_up = np.zeros_like(flux_top)

    flux_down[-1] = flux_top * tau[-1]
    num_layers = flux_down.shape[0]
    # Propagate the flux downwards from toa to boa
    for layer in reversed(range(num_layers - 1)):
        flux_down[layer] = flux_down[layer + 1] * tau[layer]

    # If albedo then reflect the flux
    flux_up[0] = flux_down[0]*albedo

    # Propagate the flux upwards from boa to toa
    for layer in range(1, num_layers):
        flux_up[layer] = flux_up[layer - 1] * tau[layer-1]

    flux_down2 = np.zeros_like(flux_down)
    flux_up2 = np.zeros_like(flux_up)

    # Propagate the flux downwards from toa to boa
    flux_down2[-1] = (
        0.5 * ((1 - tau[-1]) * (flux_up[-1] + flux_top)) + flux_top
    ) * tau[-1]

    flux_down2[-1] = (
        0.5 * ((1 - tau[-1]) * (flux_up[-1] + flux_top)) + flux_top
    ) * tau[-1]
    for layer in reversed(range(num_layers)[:-1]):
        flux_down2[layer] = (
            0.5 * ((1 - tau[layer]) * (flux_up[layer] + flux_down[layer + 1]))
            + flux_down2[layer + 1]
        ) * tau[layer]

    flux_up2[0] = albedo * flux_down2[0]
    for layer in range(num_layers)[1:]:
        flux_up2[layer] = (
            0.5 * ((1 - tau[layer - 1]) * (flux_up[layer - 1] + flux_down[layer]))
            + flux_up2[layer - 1]
        ) * tau[layer - 1]

    return flux_up2 + flux_down2
