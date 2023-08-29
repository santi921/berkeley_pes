import ijson
import numpy as np


def parse_json(json_filename, mode="normal", verbose=False):
    energies_raw, energies = [], []
    xyz_unformated, grad_unformated = [], []
    spin_list, charge_list = [], []
    atom_count, element_count, element_list = [], [], []
    element_list_single_structure = []

    with open(json_filename, "rb") as input_file:
        # load json iteratively
        parser = ijson.parse(input_file)
        ind_track, atom_count_temp = 0, 0
        trigger_count, check_ind = 0, 0
        ind_mode, ind_current = -1, -1
        line_ind = 0
        frame_count_total = 0
        for prefix, event, value in parser:
            # if "spin_multiplicity" in prefix and event == "number":
            #    print("prefix={}, event={}, value={}".format(prefix, event, value))

            if mode == "flat":
                if prefix[0:13] == "item.molecule" or prefix[0:8] == "molecule":
                    # print('prefix={}, event={}, value={}'.format(prefix, event, value))

                    if ind_mode == -1 and event == "start_array":
                        try:
                            ind_current = int(prefix.split(".")[1])
                            ind_mode = 0
                            print("entering ind mode 0")
                        except:
                            trigger = False
                            ind_mode = 1
                            print("entering ind mode 1")

                    # print('prefix={}, event={}, value={}'.format(prefix, event, value))

                    if (
                        value == None
                        and event == "null"
                        and prefix == "item.molecule.@version"
                    ):
                        trigger = True
                        trigger_count += 1

                    elif event == "number":
                        if "xyz" in prefix.split("."):
                            xyz_unformated.append(float(value))
                        if "charge" in prefix.split("."):
                            charge_list.append(float(value))

                    elif event == "string":
                        if "element" in prefix.split("."):
                            element_list.append(str(value))
                            element_list_single_structure.append(str(value))
                            if trigger:
                                atom_count.append(atom_count_temp)
                                element_count.append(
                                    len(element_list_single_structure[0])
                                )
                                atom_count_temp = 1
                                trigger = False
                                element_list_single_structure = []

                            else:
                                atom_count_temp += 1

                if value != None:
                    if "gradient" in prefix.split("."):
                        if event == "number":
                            grad_unformated.append(float(value))

                    if prefix == "item.energy":
                        if event == "number":
                            energies_raw.append(float(value))

            else:
                if value is not None:
                    if event == "string":
                        if "formula_alphabetical" in prefix.split("."):
                            # sum all integers in string
                            elements = value.split()
                            # strip elements of alphabetical characters
                            elements = [
                                element.strip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                                for element in elements
                            ]
                            # strip lowercase letters
                            elements = [
                                element.strip("abcdefghijklmnopqrstuvwxyz")
                                for element in elements
                            ]
                            # print(elements)
                            num_atoms = sum(map(int, elements))
                            element_count.append(num_atoms)

                    if event == "number":
                        if "energy_trajectory" in prefix.split("."):
                            energies_raw.append(float(value))

                        if "gradient_trajectory" in prefix.split("."):
                            # print('prefix={}, event={}, value={}'.format(prefix, event, value))
                            grad_unformated.append(float(value))

                if "molecule_trajectory" in prefix.split("."):
                    # if prefix[0:13] == "item.molecule":
                    if ind_mode == -1 and event == "start_array":
                        try:
                            ind_current = int(prefix.split(".")[1])
                            ind_mode = 0
                            print("entering ind mode 0")
                        except:
                            trigger = False
                            ind_mode = 1
                            print("entering ind mode 1")

                    if ind_mode == 0:
                        if event == "number":
                            if "xyz" in prefix.split("."):
                                xyz_unformated.append(float(value))
                            if "charge" in prefix.split("."):
                                charge_list.append(float(value))
                            if "spin_multiplicity" in prefix.split("."):
                                # print(
                                #    "prefix={}, event={}, value={}".format(
                                #        prefix, event, value
                                #    )
                                # )
                                spin_list.append(float(value))

                        if event == "string":
                            # print('prefix={}, event={}, value={}'.format(prefix, event, value))
                            if "name" in prefix.split("."):
                                element_list.append(str(value))
                                ind_current = int(prefix.split(".")[1])
                                # print(ind_current, ind_track)
                                if ind_current != ind_track:
                                    ind_track = int(ind_current)
                                    atom_count.append(atom_count_temp)
                                    atom_count_temp = 1
                                else:
                                    atom_count_temp += 1
                        if (
                            value == None
                            and event == "end_array"
                            and "molecule_trajectory" in prefix.split(".")
                            and "sites" in prefix.split(".")
                            and "species" not in prefix.split(".")
                            and "xyz" not in prefix.split(".")
                        ):
                            frame_count_total += 1

                    if ind_mode == 1:
                        if (
                            value == None
                            and event == "end_array"
                            and prefix == "item.molecule_trajectory"
                        ):
                            trigger = True
                            trigger_count += 1

                        if event == "number":
                            if "xyz" in prefix.split("."):
                                xyz_unformated.append(float(value))
                            # print('prefix={}, event={}, value={}'.format(prefix, event, value))
                            if "charge" in prefix.split("."):
                                charge_list.append(float(value))
                            if "spin_multiplicity" in prefix.split("."):
                                """print(
                                    "prefix={}, event={}, value={}".format(
                                        prefix, event, value
                                    )
                                )"""
                                spin_list.append(float(value))

                        if event == "string":
                            if "name" in prefix.split("."):
                                element_list.append(str(value))
                                if trigger:
                                    atom_count.append(atom_count_temp)
                                    atom_count_temp = 1
                                    trigger = False
                                else:
                                    atom_count_temp += 1

    if ind_mode == 0:
        atom_count.append(atom_count_temp)
        if mode == "flat":
            element_count.append(len(element_list_single_structure))
    else:
        atom_count.append(atom_count_temp)
        if mode == "flat":
            element_count.append(len(element_list_single_structure))

    if mode == "flat":
        grad_unformated = np.array(grad_unformated)
        grad_formated = grad_unformated.reshape(-1, 3)
        xyz_unformated = np.array(xyz_unformated)
        xyz_formated = xyz_unformated.reshape(-1, 3)
        atom_count = np.array(atom_count)
        frames_per_mol = atom_count / element_count
        grad_format = np.split(grad_formated, np.cumsum(atom_count)[:-1])
        xyz_format = np.split(xyz_formated, np.cumsum(atom_count)[:-1])
        element_list = np.split(element_list, np.cumsum(atom_count)[:-1])
        energies = energies_raw

    else:
        grad_unformated = np.array(grad_unformated)
        grad_formated = grad_unformated.reshape(-1, 3)
        xyz_unformated = np.array(xyz_unformated)
        xyz_formated = xyz_unformated.reshape(-1, 3)
        atom_count = np.array(atom_count)
        frames_per_mol = atom_count / element_count

        grad_format = np.split(grad_formated, np.cumsum(atom_count)[:-1])
        xyz_format = np.split(xyz_formated, np.cumsum(atom_count)[:-1])
        element_list = np.split(element_list, np.cumsum(atom_count)[:-1])  #
        # charge_list = np.split(charge_list, np.cumsum(frames_per_mol))
        # split charges into one charges per frame per molecule

        # split energies into frames per molecule
        running_start = 0
        charges = []
        spins = []
        for i in range(len(frames_per_mol)):
            energies.append(
                energies_raw[running_start : running_start + int(frames_per_mol[i])]
            )
            charges.append(
                charge_list[running_start : running_start + int(frames_per_mol[i])]
            )
            spins.append(
                spin_list[running_start : running_start + int(frames_per_mol[i])]
            )
            running_start += int(frames_per_mol[i])

        xyz_format = [
            array.reshape(int(frames_per_mol[ind_frame]), element_count[ind_frame], 3)
            for ind_frame, array in enumerate(xyz_format)
        ]
        grad_format = [
            array.reshape(int(frames_per_mol[ind_frame]), element_count[ind_frame], 3)
            for ind_frame, array in enumerate(grad_format)
        ]
        element_list = [
            array.reshape(int(frames_per_mol[ind_frame]), element_count[ind_frame])
            for ind_frame, array in enumerate(element_list)
        ]

    composition_list = []
    for elements_repeated in element_list:
        composition = {}
        for element in elements_repeated[0]:
            if element in composition:
                composition[element] += 1
            else:
                composition[element] = 1
        # sort composition by key
        composition = dict(sorted(composition.items()))
        composition_list.append(composition)

    if verbose:
        print("element_list len:     {}".format(len(element_list)))
        print("element_count len:    {}".format(len(element_count)))
        print("atom_count len:       {}".format(len(atom_count)))
        print("spins len:            {}".format(len(spins)))
        print("xyz_format len:       {}".format(len(xyz_format)))
        print("energies len:         {}".format(len(energies)))
        print("grad_format len:      {}".format(len(grad_format)))
        print("composition_list len: {}".format(len(composition_list)))
        print("charge_list len:      {}".format(len(charges)))
        print("frames total:         {}".format(frame_count_total))
        print("frames per mol:       {}".format(frames_per_mol))
        print("sum frames per mol:   {}".format(np.sum(frames_per_mol)))

    data = {
        "energies": energies,
        "grads": grad_format,
        "xyz": xyz_format,
        "elements": element_list,
        "frames_per_mol": frames_per_mol,
        "atom_count": atom_count,
        "element_composition": composition_list,
        "charges": charges,
        "spin": spins,
    }
    return data
