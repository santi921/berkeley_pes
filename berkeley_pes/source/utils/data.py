"""
    Functions for converting data from one format to another
"""
import os
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
    "Br": 35,
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
            f.write(
                'Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc="F F F"\n'.format(
                    energy, energy
                )
            )
            for site in sites:
                # write in format C        0.00000000       0.00000000       0.66748000       0.00000000       0.00000000      -5.01319916
                f.write(
                    "{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(
                        site["name"].sljust(0),
                        site["xyz"][0],
                        site["xyz"][1],
                        site["xyz"][2],
                        site["force"][0],
                        site["force"][1],
                        site["force"][2],
                    )
                )
                # f.write("{:.8}\t {:.8}\t {:.8}\t {:.8}\t {:.8}\t {:.8}\t {:.8}\n".format(site["name"], site["xyz"][0], site["xyz"][1], site["xyz"][2], site["force"][0], site["force"][1], site["force"][2]))


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
            # row = df.iloc[i]

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

    np.savez(
        file_out,
        E=np.array(energies),
        F=np.array(forces_list),
        R=np.array(positions_list),
        z=np.array(atomic_numbers),
    )


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
        for ind_frame, (
            energies_frame,
            grads_frame,
            xyzs_frame,
            elements_frame,
        ) in enumerate(zip(energies, grads_list, xyzs_list, elements_list)):
            # print(ind_frame, len(energies_frame))
            for ind_mol, (energy, grad, xyz, elements) in enumerate(
                zip(energies_frame, grads_frame, xyzs_frame, elements_frame)
            ):
                frame_count_global += 1
                n_atoms = len(elements)
                # print(elements)
                f.write(str(n_atoms) + "\n")
                f.write(
                    'Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc="F F F"\n'.format(
                        energy, energy
                    )
                )
                for ind_atom, (xyz, grad) in enumerate(zip(xyz, grad)):
                    # print(elements[0])
                    f.write(
                        "{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(
                            elements[ind_atom],
                            xyz[0],
                            xyz[1],
                            xyz[2],
                            grad[0],
                            grad[1],
                            grad[2],
                        )
                    )

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
        for ind_frame, (energy, grad, xyz, elements) in enumerate(
            zip(energies, grads_list, xyzs_list, elements_list)
        ):
            frame_count_global += 1
            n_atoms = len(elements)
            # print(elements)
            f.write(str(n_atoms) + "\n")
            f.write(
                'Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc="F F F"\n'.format(
                    energy, energy
                )
            )
            for ind_atom, (xyz, grad) in enumerate(zip(xyz, grad)):
                # print(elements[0])
                f.write(
                    "{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(
                        elements[ind_atom],
                        xyz[0],
                        xyz[1],
                        xyz[2],
                        grad[0],
                        grad[1],
                        grad[2],
                    )
                )

    print("Wrote {} frames to {}".format(frame_count_global, file_out))


def write_dict_to_ase_trajectory(
    dict_info, file_out, separate_charges=False, separate_composition=False
):
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
        for ind_frame, (
            energies_frame,
            grads_frame,
            xyzs_frame,
            elements_frame,
        ) in enumerate(zip(energies, grads_list, xyzs_list, elements_list)):
            # print(ind_frame, len(energies_frame))
            for ind_mol, (energy, grad, xyz, elements) in enumerate(
                zip(energies_frame, grads_frame, xyzs_frame, elements_frame)
            ):
                frame_count_global += 1
                n_atoms = len(elements)
                # print(elements)
                f.write(str(n_atoms) + "\n")
                f.write(
                    'Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc="F F F"\n'.format(
                        energy, energy
                    )
                )
                for ind_atom, (xyz, grad) in enumerate(zip(xyz, grad)):
                    # print(elements[0])
                    f.write(
                        "{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(
                            elements[ind_atom],
                            xyz[0],
                            xyz[1],
                            xyz[2],
                            grad[0],
                            grad[1],
                            grad[2],
                        )
                    )

    print("Wrote {} frames to {}".format(frame_count_global, file_out))


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


def write_ase(f, elements, energy, gradient, xyzs):
    n_atoms = len(elements)
    f.write(str(n_atoms) + "\n")
    f.write(
        'Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc="F F F "\n'.format(
            energy, energy
        )
    )
    for ind_atom, (element, grad, xyz) in enumerate(zip(elements, gradient, xyzs)):
        f.write(
            "{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(
                element,
                xyz[0],
                xyz[1],
                xyz[2],
                grad[0],
                grad[1],
                grad[2],
            )
        )


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


def write_dict_to_ase_trajectory(
    dict_info,
    root_out,
    separate_charges=False,
    separate_composition=False,
    separate_spin=False,
):
    """
    Takes a dictionary organized by trajectories and writes it to an ase file
    """

    energies = dict_info["energies"]
    grads_list = dict_info["grads"]
    xyzs_list = dict_info["xyz"]
    elements_list = dict_info["elements"]
    composition_list = dict_info["element_composition"]
    charge_list = dict_info["charges"]
    charge_list_single = [i[0] for i in charge_list]
    list_charge_unique = np.unique(charge_list_single)

    if separate_spin:
        spin_list = dict_info["spin"]
        spin_list_single = [i[0] for i in spin_list]
        list_spin_unique = np.unique(spin_list_single)

    if separate_spin and separate_composition and separate_charges:
        separate_charge_comp_dict = separate_into_charge_and_comp_and_spin(dict_info)
    elif separate_composition and separate_charges:
        separate_charge_comp_dict = separate_into_charge_and_comp(dict_info)
    elif separate_composition:
        separate_charge_comp_dict = separate_into_comp(dict_info)
    else:
        separate_charge_comp_dict = None

    # create a folder for each charge
    frame_count_global = 0

    if separate_charges and separate_spin:
        for charge in list_charge_unique:
            frame_count_charge = 0
            if not os.path.exists(
                root_out + "/charge_" + str(charge)
            ) and not os.path.exists(root_out + "/charge_" + "neg_" + str(charge)):
                charge_temp = int(charge)
                if charge < 0:
                    charge_temp = "neg_" + str(abs(charge_temp))
                os.makedirs(root_out + "/charge_" + str(charge_temp), exist_ok=True)

            for spin in list_spin_unique:
                if not os.path.exists(
                    root_out + "/charge_" + str(charge_temp) + "/spin_" + str(int(spin))
                ):
                    os.makedirs(
                        root_out
                        + "/charge_"
                        + str(charge_temp)
                        + "/spin_"
                        + str(int(spin)),
                        exist_ok=True,
                    )

            for spin_key, dict_info_temp in separate_charge_comp_dict[charge].items():
                for comp_key, dict_info in dict_info_temp.items():
                    comp_count = 0
                    file = (
                        root_out
                        + "/charge_"
                        + str(charge_temp)
                        + "/spin_"
                        + str(int(spin_key))
                        + "/{}.xyz".format(comp_key)
                    )
                    with open(file, "w") as f:
                        for ind_frame, (
                            energies_frame,
                            grads_frame,
                            xyzs_frame,
                            elements_frame,
                        ) in enumerate(
                            zip(
                                dict_info["energies"],
                                dict_info["grads"],
                                dict_info["xyzs"],
                                dict_info["elements"],
                            )
                        ):
                            for ind_mol, (energy, grad, xyz, elements) in enumerate(
                                zip(
                                    energies_frame,
                                    grads_frame,
                                    xyzs_frame,
                                    elements_frame,
                                )
                            ):
                                frame_count_global += 1
                                write_ase(f, elements, energy, grad, xyz)
                                comp_count += 1
                                frame_count_charge += 1
                                frame_count_global += 1

                    print(
                        "frames to {}: \t\t {}".format(
                            file.split("/")[-1].split(".")[0], comp_count
                        )
                    )
            print(
                "frames charge {} folder:\t {}".format(charge_temp, frame_count_charge)
            )
        print("frames total: \t\t {}".format(frame_count_global))

    elif separate_charges:
        for charge in list_charge_unique:
            frame_count_charge = 0
            if not os.path.exists(root_out + str(charge)):
                charge_temp = int(charge)
                if charge < 0:
                    charge_temp = "neg_" + str(abs(charge_temp))
                os.makedirs(root_out + str(charge_temp), exist_ok=True)

            for comp_key, dict_info in separate_charge_comp_dict[charge].items():
                comp_count = 0
                file = root_out + str(charge_temp) + "/{}.xyz".format(comp_key)
                with open(file, "w") as f:
                    for ind_frame, (
                        energies_frame,
                        grads_frame,
                        xyzs_frame,
                        elements_frame,
                    ) in enumerate(
                        zip(
                            dict_info["energies"],
                            dict_info["grads"],
                            dict_info["xyzs"],
                            dict_info["elements"],
                        )
                    ):
                        for ind_mol, (energy, grad, xyz, elements) in enumerate(
                            zip(
                                energies_frame,
                                grads_frame,
                                xyzs_frame,
                                elements_frame,
                            )
                        ):
                            frame_count_global += 1
                            write_ase(f, elements, energy, grad, xyz)
                            comp_count += 1
                            frame_count_charge += 1
                            frame_count_global += 1

                print(
                    "frames to {}: \t\t {}".format(
                        file.split("/")[-1].split(".")[0], comp_count
                    )
                )

            print(
                "frames charge {} folder:\t {}".format(charge_temp, frame_count_charge)
            )
        print("frames total: \t\t {}".format(frame_count_global))

    else:
        print("writing as a single file")
        if not os.path.exists(root_out):
            # make it if it doesn't
            os.makedirs(root_out, exist_ok=True)

        with open(root_out + "combined.xyz", "w") as f:
            for ind_frame, (
                energies_frame,
                grads_frame,
                xyzs_frame,
                elements_frame,
            ) in enumerate(zip(energies, grads_list, xyzs_list, elements_list)):
                # print(ind_frame, len(energies_frame))
                for ind_mol, (energy, grad, xyz, elements) in enumerate(
                    zip(energies_frame, grads_frame, xyzs_frame, elements_frame)
                ):
                    frame_count_global += 1
                    n_atoms = len(elements)
                    # print(elements)
                    f.write(str(n_atoms) + "\n")
                    f.write(
                        'Properties=species:S:1:pos:R:3:forces:R:3 energy={} free_energy={} pbc="F F F"\n'.format(
                            energy, energy
                        )
                    )
                    for ind_atom, (xyz, grad) in enumerate(zip(xyz, grad)):
                        # print(elements[0])
                        f.write(
                            "{:2} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8} {:15.8}\n".format(
                                elements[ind_atom],
                                xyz[0],
                                xyz[1],
                                xyz[2],
                                grad[0],
                                grad[1],
                                grad[2],
                            )
                        )
        print("Wrote {} frames to {}".format(frame_count_global, root_out))
