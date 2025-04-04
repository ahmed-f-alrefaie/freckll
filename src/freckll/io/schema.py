from pydantic import BaseModel, FilePath
from astropy import units as u

class PlanetSchema(BaseModel):
    """ Schema for the planet model."""
    mass: u.Quantity
    radius: u.Quantity
    distance: u.Quantity = None


class StarSpectraSchema(BaseModel):
    pass
