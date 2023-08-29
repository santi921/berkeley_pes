import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from berkeley_pes.source.utils.constants import element_to_number
from berkeley_pes.source.utils.data import traj_to_dict
from berkeley_pes.source.utils.data import (
    separate_into_charge_and_comp_and_spin,
    separate_into_charge_and_comp,
    separate_into_comp,
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
    dict_info,
    root_out,
    separate_charges=False,
    separate_composition=False,
    separate_spin=False,
):
    """
    Takes a dictionary organized by trajectories and writes it to an ase file
    Takes:
        dict_info: dictionary with info in order as it's read in from the json file
        root_out: root directory to write to
        separate_charges: whether to separate by charges
        separate_composition: whether to separate by composition
        separate_spin: whether to separate by spin
    Returns:
        None
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


def write_dict_to_ase_trajectory(
    dict_info,
    root_out,
    separate_charges=False,
    separate_composition=False,
    separate_spin=False,
):
    """
    Takes a dictionary organized by trajectories and writes it to an ase file
    Takes:
        dict_info: dictionary with info in order as it's read in from the json file
        root_out: root directory to write to
        separate_charges: whether to separate by charges
        separate_composition: whether to separate by composition
        separate_spin: whether to separate by spin
    Returns:
        None
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
