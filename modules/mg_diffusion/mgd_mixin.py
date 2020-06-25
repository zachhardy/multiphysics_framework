
from material import NeutronicsMaterial

class MGDMixin:
  
  def _parse_materials(self):
    """ Get neutronics properties and sort by zone. """
    materials = []
    for material in self.problem.materials:
        if isinstance(material, NeutronicsMaterial):
                assert material.G == self.G, (
                    "Incompatible group structures."
                )
                materials += [material]
    materials.sort(key=lambda x: x.material_id)
    return materials