"""
    Functions for converting data from one format to another
"""

import numpy as np 
import pandas as pd
from tqdm import tqdm


element_to_number = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35
}


def traj_to_dict(df_row): 
    """
    Converts dataframe info into a single dictionary to write
    Takes: 
        df_row: a single row of the dataframe
    """
    
    energies = df_row["energy_trajectory"]
    gradients = df_row["gradient_trajectory"]
    sites = df_row["molecule_trajectory"]
    ret_list = []
    items = len(energies)
    n_gradients = len(gradients)

    assert items == n_gradients, "Number of energies and gradients must be equal"
    benchmark_gradient_len = len(gradients[0])
    
    for frame in range(items): 
        assert len(gradients[frame]) == benchmark_gradient_len, "Number of gradients must be equal for all energies - did an atom disappear!?"
        gradient_len = len(gradients[frame])
        sites = df_row["molecule_trajectory"][frame]["sites"]
        assert len(sites) == gradient_len, "Number of sites must be equal to number of gradients"
        for i in range(gradient_len): 
            sites[i]["force"] = gradients[frame][i]

        ret_list.append({"energy": energies[frame], "sites": sites})
    return ret_list


def write_to_npz(list_dict, file_out): 
    """
    Takes a dictionary of the form: 
    {
        "energy": 0.0,
        "sites": [
            {
                "force": [0.0, 0.0, 0.0],
                "name": "C",
                "xyz": [0.0, 0.0, 0.0]
            }
        ]
    }
    and writes it to a npz file of the form: 
    dict = {
        "E": np.array([0.0]),
        "F": np.array([[0.0, 0.0, 0.0]]),
        "R": np.array([[0.0, 0.0, 0.0]]),
        "z": np.array([6])
    """
    
    energies = []
    forces_list = []
    positions_list = []

    for dict in list_dict:
        energy = dict["energy"]
        sites = dict["sites"]
        forces = []
        positions = []
        atomic_numbers = []
        for site in sites: 
            forces.append(site["force"])
            positions.append(site["xyz"])
            atomic_numbers.append(element_to_number[site["name"]])
        
        energies.append(energy)
        forces_list.append(forces)
        positions_list.append(positions)
    
    np.savez(file_out, 
             E=np.array(energies), 
             F=np.array(forces_list), 
             R=np.array(positions_list), 
             z=np.array(atomic_numbers))


def write_to_ase(list_dict, file_out): 
    """
    Takes list of dictionaries of the form 
    {
        "energy": 0.0,
        "sites": [
            {
                "force": [0.0, 0.0, 0.0],
                "name": "C",
                "xyz": [0.0, 0.0, 0.0]
            }
        ]
    }
    and writes to ase file 
    """
    with open(file_out, "w") as f:
        for dict in list_dict:
            energy = dict["energy"]
            sites = dict["sites"]
            forces = []
            positions = []
            atomic_numbers = []
            for site in sites: 
                forces.append(site["force"])
                positions.append(site["xyz"])
                atomic_numbers.append(element_to_number[site["name"]])
            n_atoms = len(atomic_numbers)

            f.write(str(n_atoms) + "\n")
            f.write("Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc=\"F F F\"\n".format(energy, energy))
            for site in sites:
                # write in format C        0.00000000       0.00000000       0.66748000       0.00000000       0.00000000      -5.01319916    
                f.write("{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(site["name"].sljust(0), site["xyz"][0], site["xyz"][1], site["xyz"][2], site["force"][0], site["force"][1], site["force"][2]))
                #f.write("{:.8}\t {:.8}\t {:.8}\t {:.8}\t {:.8}\t {:.8}\t {:.8}\n".format(site["name"], site["xyz"][0], site["xyz"][1], site["xyz"][2], site["force"][0], site["force"][1], site["force"][2]))


def write_whole_df_to_npz(df_file, file_out): 
    """
    Takes a dataframe and writes it to a npz file
    """
    chunks = pd.read_json(df_file, chunksize=100, lines=True)
    print("done loading json")
    energies = []
    forces_list = []
    positions_list = []
    
    for chunk in tqdm(chunks):
        for row in chunk.iterrows():
            #row = df.iloc[i]

            list_dict = traj_to_dict(row)

            for dict in list_dict:
                energy = dict["energy"]
                sites = dict["sites"]
                forces = []
                positions = []
                atomic_numbers = []
                for site in sites: 
                    forces.append(site["force"])
                    positions.append(site["xyz"])
                    atomic_numbers.append(element_to_number[site["name"]])
                
                energies.append(energy)
                forces_list.append(forces)
                positions_list.append(positions)
        
    np.savez(file_out, 
             E=np.array(energies), 
             F=np.array(forces_list), 
             R=np.array(positions_list), 
             z=np.array(atomic_numbers))

def write_dictionary_to_ase_xyz(dict_info, file_out): 
    """
    Takes a dictionary and writes it to a ase_xyz file
    """ 
    energies = dict_info["energies"]
    grads_list = dict_info["grads"]
    xyzs_list = dict_info["xyz"]
    elements_list = dict_info["elements"]
    frame_count_global = 0 
    with open(file_out, "w") as f:    
        for ind_frame, (energies_frame, grads_frame, xyzs_frame, elements_frame) in enumerate(zip(energies, grads_list, xyzs_list, elements_list)):
            #print(ind_frame, len(energies_frame))
            for ind_mol, (energy, grad, xyz, elements) in enumerate(zip(energies_frame, grads_frame, xyzs_frame, elements_frame)):
                frame_count_global += 1
                n_atoms = len(elements)
                #print(elements)
                f.write(str(n_atoms) + "\n")
                f.write("Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc=\"F F F\"\n".format(energy, energy))
                for ind_atom, (xyz, grad) in enumerate(zip(xyz, grad)): 
                    #print(elements[0])
                    f.write("{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(elements[ind_atom], xyz[0], xyz[1], xyz[2], grad[0], grad[1], grad[2]))

    
    print("Wrote {} frames to {}".format(frame_count_global, file_out))


def write_dict_to_ase_single_mol(dict_info, file_out): 
    """
    Takes a dictionary with a single molecule/gradient/energy and writes it to an ase file
    """
    energies = dict_info["energies"]
    grads_list = dict_info["grads"]
    xyzs_list = dict_info["xyz"]
    elements_list = dict_info["elements"]
    frame_count_global = 0 
    with open(file_out, "w") as f:    
        for ind_frame, (energy, grad, xyz, elements) in enumerate(zip(energies, grads_list, xyzs_list, elements_list)):
            frame_count_global += 1
            n_atoms = len(elements)
            #print(elements)
            f.write(str(n_atoms) + "\n")
            f.write("Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc=\"F F F\"\n".format(energy, energy))
            for ind_atom, (xyz, grad) in enumerate(zip(xyz, grad)): 
                #print(elements[0])
                f.write("{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(elements[ind_atom], xyz[0], xyz[1], xyz[2], grad[0], grad[1], grad[2]))

    
    print("Wrote {} frames to {}".format(frame_count_global, file_out))


def write_dict_to_ase_trajectory(dict_info, file_out, separate_charges=False, separate_composition=False): 
    """
    Takes a dictionary organized by trajectories and writes it to an ase file
    """
    # TODO: chunk by charges
    # TODO: chunk by atomic composition
    
    energies = dict_info["energies"]
    grads_list = dict_info["grads"]
    xyzs_list = dict_info["xyz"]
    elements_list = dict_info["elements"]
    frame_count_global = 0 
    with open(file_out, "w") as f:    
        for ind_frame, (energies_frame, grads_frame, xyzs_frame, elements_frame) in enumerate(zip(energies, grads_list, xyzs_list, elements_list)):
            #print(ind_frame, len(energies_frame))
            for ind_mol, (energy, grad, xyz, elements) in enumerate(zip(energies_frame, grads_frame, xyzs_frame, elements_frame)):
                frame_count_global += 1
                n_atoms = len(elements)
                #print(elements)
                f.write(str(n_atoms) + "\n")
                f.write("Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc=\"F F F\"\n".format(energy, energy))
                for ind_atom, (xyz, grad) in enumerate(zip(xyz, grad)): 
                    #print(elements[0])
                    f.write("{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(elements[ind_atom], xyz[0], xyz[1], xyz[2], grad[0], grad[1], grad[2]))

    
    print("Wrote {} frames to {}".format(frame_count_global, file_out))

