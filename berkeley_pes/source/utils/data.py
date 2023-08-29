"""
    Functions for converting data from one format to another
"""
import numpy as np


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
        assert (
            len(gradients[frame]) == benchmark_gradient_len
        ), "Number of gradients must be equal for all energies - did an atom disappear!?"
        gradient_len = len(gradients[frame])
        sites = df_row["molecule_trajectory"][frame]["sites"]
        assert (
            len(sites) == gradient_len
        ), "Number of sites must be equal to number of gradients"
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

    np.savez(
        file_out,
        E=np.array(energies),
        F=np.array(forces_list),
        R=np.array(positions_list),
        z=np.array(atomic_numbers),
    )


def comp_dict_to_string(i):
    comp_string = ""
    for key, value in sorted(i.items()):
        comp_string = comp_string + key + str(value) + "_"
    comp_string = comp_string[:-1]
    return comp_string


def separate_into_charge_and_comp(dict_info):
    energies = dict_info["energies"]
    grads_list = dict_info["grads"]
    xyzs_list = dict_info["xyz"]
    elements_list = dict_info["elements"]
    composition_list = dict_info["element_composition"]
    charge_list = dict_info["charges"]
    charge_list_single = [i[0] for i in charge_list]
    list_charge_unique = np.unique(charge_list_single)

    separate_charge_comp_dict = (
        {}
    )  # dict with charges and info for each frame in composition

    for charge in list_charge_unique:  # instantiate
        separate_charge_comp_dict[charge] = {}

    comp_string_list = []

    for ind, i in enumerate(composition_list):
        comp_string = comp_dict_to_string(i)
        comp_string_list.append(comp_string)
        charge = charge_list[ind][0]

        if comp_string not in separate_charge_comp_dict[charge].keys():
            separate_charge_comp_dict[charge][comp_string] = {
                "energies": [],
                "grads": [],
                "xyzs": [],
                "elements": [],
                "charge": [],
            }

        separate_charge_comp_dict[charge][comp_string]["energies"].append(energies[ind])
        separate_charge_comp_dict[charge][comp_string]["grads"].append(grads_list[ind])
        separate_charge_comp_dict[charge][comp_string]["xyzs"].append(xyzs_list[ind])
        separate_charge_comp_dict[charge][comp_string]["elements"].append(
            elements_list[ind]
        )

    return separate_charge_comp_dict


def separate_into_comp(dict_info):
    energies = dict_info["energies"]
    grads_list = dict_info["grads"]
    xyzs_list = dict_info["xyz"]
    elements_list = dict_info["elements"]
    composition_list = dict_info["element_composition"]
    charge_list = dict_info["charges"]
    charge_list_single = [i[0] for i in charge_list]

    separate_charge_comp_dict = (
        {}
    )  # dict with charges and info for each frame in composition

    # for charge in list_charge_unique:  # instantiate
    #    separate_charge_comp_dict[charge] = {}

    comp_string_list = []
    for ind, i in enumerate(composition_list):
        comp_string = comp_dict_to_string(i)
        comp_string_list.append(comp_string)

        if comp_string not in separate_charge_comp_dict.keys():
            separate_charge_comp_dict[comp_string] = {
                "energies": [],
                "grads": [],
                "xyzs": [],
                "elements": [],
            }

        separate_charge_comp_dict[comp_string]["energies"].append(energies[ind])
        separate_charge_comp_dict[comp_string]["grads"].append(grads_list[ind])
        separate_charge_comp_dict[comp_string]["xyzs"].append(xyzs_list[ind])
        separate_charge_comp_dict[comp_string]["elements"].append(elements_list[ind])

    return separate_charge_comp_dict


def separate_into_charge_and_comp_and_spin(dict_info):
    energies = dict_info["energies"]
    grads_list = dict_info["grads"]
    spin_list = dict_info["spin"]
    charge_list = dict_info["charges"]
    xyzs_list = dict_info["xyz"]
    elements_list = dict_info["elements"]
    composition_list = dict_info["element_composition"]
    charge_list_single = [i[0] for i in charge_list]
    spin_list_single = [i[0] for i in spin_list]
    list_charge_unique = np.unique(charge_list_single)
    list_spin_unique = np.unique(spin_list_single)

    separate_charge_comp_dict = (
        {}
    )  # dict with charges and info for each frame in composition

    for charge in list_charge_unique:  # instantiate
        separate_charge_comp_dict[charge] = {}
        for spin in list_spin_unique:
            separate_charge_comp_dict[charge][spin] = {}

    comp_string_list = []
    print(separate_charge_comp_dict)
    for ind, i in enumerate(composition_list):
        comp_string = comp_dict_to_string(i)
        comp_string_list.append(comp_string)
        charge = charge_list[ind][0]
        spin = spin_list[ind][0]

        if comp_string not in separate_charge_comp_dict[charge][spin].keys():
            separate_charge_comp_dict[charge][spin][comp_string] = {
                "energies": [],
                "grads": [],
                "xyzs": [],
                "elements": [],
                "charge": [],
                "spin": [],
            }

        separate_charge_comp_dict[charge][spin][comp_string]["energies"].append(
            energies[ind]
        )
        separate_charge_comp_dict[charge][spin][comp_string]["grads"].append(
            grads_list[ind]
        )
        separate_charge_comp_dict[charge][spin][comp_string]["xyzs"].append(
            xyzs_list[ind]
        )
        separate_charge_comp_dict[charge][spin][comp_string]["elements"].append(
            elements_list[ind]
        )
        separate_charge_comp_dict[charge][spin][comp_string]["charge"].append(
            charge_list[ind]
        )
        separate_charge_comp_dict[charge][spin][comp_string]["spin"].append(
            spin_list[ind]
        )

    return separate_charge_comp_dict
