import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




mm = 1/1000.
cm = 1/100.
um = 1/1000. * 1/1000.

allowed_materials_nuclear_radiation_lengths = {
    'G4_Cu' : 0.0143558, # Copper
    'G4_Pb': 0.00561253, # Lead
    'stainless_steel': 0.0174238, # Stainless steel
    'G4_W': 0.00350418, # Tungsten: ,
    'G4_Si': 0.0936607, # Silicone
    'G4_Mo' : 15.25 * cm,
}

allowed_materials_nuclear_interaction_lengths = {
    'G4_Cu' : 0.155879, # Copper
    'G4_Pb': 0.182479, # Lead
    'stainless_steel': 0.16588, # Stainless steel
    'G4_W': 0.103116, # Tungsten: ,
    'G4_Si': 0.456603, # Silicone
    'G4_Mo' : 0.9593 * cm,
}

allowed_active_length = {
    0.0002, 0.0015, 0.0012
}



def draw_box(ax, dx, dy, dz, z_center):
    # Define the vertices of the box
    x = [-dx / 2, dx / 2, dx / 2, -dx / 2, -dx / 2, dx / 2, dx / 2, -dx / 2]
    y = [-dy / 2, -dy / 2, dy / 2, dy / 2, -dy / 2, -dy / 2, dy / 2, dy / 2]
    z = [z_center - dz / 2] * 4 + [z_center + dz / 2] * 4

    # Define the 6 faces of the box
    vertices = [[(z[i], x[i], y[i]) for i in [0, 1, 2, 3]],
                [(z[i], x[i], y[i]) for i in [4, 5, 6, 7]],
                [(z[i], x[i], y[i]) for i in [0, 3, 7, 4]],
                [(z[i], x[i], y[i]) for i in [1, 2, 6, 5]],
                [(z[i], x[i], y[i]) for i in [0, 1, 5, 4]],
                [(z[i], x[i], y[i]) for i in [2, 3, 7, 6]]]

    # Create a Poly3DCollection from the vertices
    box = Poly3DCollection(vertices, alpha=0.25, linewidths=1, edgecolors='r')
    ax.add_collection3d(box)


class HetrogeneousSamplingCalorimeter:
    def __init__(self, dx: float, dy: float):
        self.layers = []
        self.dx = dx
        self.dy = dy
        self.z_center = 0.0
        self.layer_number = 0

        self.pre_absorber_rad_lengths = 0.0
        self.pre_absorber_nuclear_interaction_lengths = 0.0
        self.pre_absorber_width = 0.0

    def add_layer(self, material: str, dz: float):
        assert material in allowed_materials_nuclear_interaction_lengths
        active = True if material == 'G4_Si' else False
        data = {'material': material, 'dz': dz, 'dx': self.dx, 'dy': self.dy, 'active': active,
                'z_center': self.z_center + dz / 2, 'limits': {
                'max_step_length': dz / 4,
                'minimum_kinetic_energy': 0.01e-9,

            }, 'layer_number': self.layer_number}
        self.layer_number += 1

        if not active:
            self.pre_absorber_rad_lengths += dz / allowed_materials_nuclear_radiation_lengths[material]
            self.pre_absorber_nuclear_interaction_lengths += dz / allowed_materials_nuclear_interaction_lengths[material]
            self.pre_absorber_width += dz

        if active:
            # data['layer_number'] = self.layer_number
            data['pre_absorber_nuclear_interaction_lengths'] = self.pre_absorber_nuclear_interaction_lengths
            data['pre_absorber_rad_lengths'] = self.pre_absorber_rad_lengths
            data['pre_absorber_width'] = self.pre_absorber_width
            data['sensor_nuclear_interaction_lengths'] = dz / allowed_materials_nuclear_radiation_lengths[material]
            data['sensor_rad_lengths'] = dz / allowed_materials_nuclear_interaction_lengths[material]


            self.pre_absorber_rad_lengths = 0.0
            self.pre_absorber_nuclear_interaction_lengths = 0.0
            self.pre_absorber_width = 0.0

        self.layers.append(data)
        self.z_center += dz


    def visualize(self):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for layer in self.layers:
            draw_box(ax,  layer['dx'],  layer['dy'], layer['dz'], layer['z_center'])

        # Set labels
        ax.set_xlabel('Z (m)')
        ax.set_ylabel('X (m)')
        ax.set_zlabel('Y (m)')

        ax.set_xlim([-2, self.z_center + 2])
        ax.set_ylim([-self.dx / 2 - 2, self.dx / 2 + 2])
        ax.set_zlim([-self.dy / 2 - 2, self.dy / 2 + 2])

        # Show plot
        plt.show()

    def get_design_for_g4_construction(self, to_json=True):
        output = {
            'layers': self.layers,
            'worldSizeX': self.dx * 1.2,
            'worldSizeY': self.dy * 1.2,
            'worldSizeZ': self.z_center * 2.1,
            'worldPositionX': 0,
            'worldPositionY': 0,
            'worldPositionZ': 0,
            'type': 0,
            'limits' : {
                'max_step_length': 0.001,
                'minimum_kinetic_energy' : 0.05,

            }
        }
        return output




def get_ref_alpha_design():
    calo = HetrogeneousSamplingCalorimeter(dx=1.5, dy=1.5)
    for i in range(14):
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('G4_Cu', 6 * mm)
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('G4_Pb', 2.1 * mm)
        calo.add_layer('stainless_steel', 0.6 * mm)
        calo.add_layer('G4_Cu', 0.1 * mm)

    for i in range(12):
        calo.add_layer('stainless_steel', 3.5 * cm)
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('G4_Cu', 6. * mm)

    for i in range(16):
        calo.add_layer('stainless_steel', 6.8 * cm)
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('G4_Cu', 6. * mm)

    # calo.visualize()
    design = calo.get_design_for_g4_construction()
    return design


def get_ref_beta_design():
    calo = HetrogeneousSamplingCalorimeter(dx=1.5, dy=1.5)

    calo.add_layer('stainless_steel', 1 * mm)
    for i in range(14):
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('stainless_steel', 6 * mm)
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('stainless_steel', 2.1 * mm)
        calo.add_layer('stainless_steel', 0.6 * mm)
        calo.add_layer('stainless_steel', 0.1 * mm)

    for i in range(12):
        calo.add_layer('stainless_steel', 3.5 * cm)
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('stainless_steel', 6. * mm)

    for i in range(16):
        calo.add_layer('stainless_steel', 6.8 * cm)
        calo.add_layer('G4_Si', 200 * um)
        calo.add_layer('stainless_steel', 6. * mm)

    # calo.visualize()
    design = calo.get_design_for_g4_construction()
    return design

def get_ref_gamma_design():
    calo = HetrogeneousSamplingCalorimeter(dx=1.5, dy=1.5)

    calo.add_layer('stainless_steel', 1 * mm)
    for i in range(14):
        calo.add_layer('G4_Si', 60 * um)
        calo.add_layer('stainless_steel', 6 * mm)
        # calo.add_layer('G4_Si', 120 * um)
        calo.add_layer('stainless_steel', 2.1 * mm)
        calo.add_layer('stainless_steel', 0.6 * mm)
        calo.add_layer('stainless_steel', 0.1 * mm)

    for i in range(12):
        calo.add_layer('stainless_steel', 3.5 * cm)
        calo.add_layer('G4_Si', 60 * um)
        calo.add_layer('stainless_steel', 6. * mm)

    for i in range(16):
        calo.add_layer('stainless_steel', 6.8 * cm)
        calo.add_layer('G4_Si', 60 * um)
        calo.add_layer('stainless_steel', 6. * mm)

    # calo.visualize()
    design = calo.get_design_for_g4_construction()
    return design

def get_ref_slab_design():
    calo = HetrogeneousSamplingCalorimeter(dx=1.5, dy=1.5)

    # calo.add_layer('stainless_steel', 1 * mm)
    # for i in range(14):
    #     calo.add_layer('G4_Si', 60 * um)
    #     calo.add_layer('stainless_steel', 6 * mm)
    #     # calo.add_layer('G4_Si', 120 * um)
    #     calo.add_layer('stainless_steel', 2.1 * mm)
    #     calo.add_layer('stainless_steel', 0.6 * mm)
    #     calo.add_layer('stainless_steel', 0.1 * mm)
    #
    # for i in range(12):
    #     calo.add_layer('stainless_steel', 3.5 * cm)
    #     calo.add_layer('G4_Si', 60 * um)
    #     calo.add_layer('stainless_steel', 6. * mm)
    #
    # for i in range(16):
    #     calo.add_layer('stainless_steel', 6.8 * cm)
    #     calo.add_layer('G4_Si', 60 * um)
    calo.add_layer('stainless_steel', 2000. * mm)

    # calo.visualize()
    design = calo.get_design_for_g4_construction()
    return design

if __name__ == '__main__':
    design = get_ref_alpha_design()
    with open('design_alpha.json', 'w') as outfile:
        json.dump(design, outfile)