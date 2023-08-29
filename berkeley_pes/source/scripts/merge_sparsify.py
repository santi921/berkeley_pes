import glob
import json
import os
from ase.io import read


def merge_xyz_files(file_list, file_out):
    """
    merge a list of xyz files into one xyz file
    """
    with open(file_out, "w") as f:
        for file in file_list:
            with open(file, "r") as f_temp:
                for line in f_temp:
                    f.write(line)


def main():
    # read json with options
    options = json.load(open("./options.json"))
    # read xyz file
    root_folder = options["root_folder"]

    # using glob to get all xyz files in the folder

    print("root folder: {}".format(root_folder))
    files_target = glob.glob(root_folder + "/*/*/*_sparse.xyz")
    # print(files_target)
    print("Number of files:        {}".format(len(files_target)))

    # organize into list of list where each sublist is a folder
    dict_list = {}
    total_count = 0
    for file in files_target:
        composition = file.split("/")[-1].split("_sparse")[0]
        charge = file.split("/")[-3]
        spin = file.split("/")[-2]
        # print("charge: {}, spin: {}".format(charge, spin))
        if composition not in dict_list.keys():
            dict_list[composition] = {}

        if charge not in dict_list[composition].keys():
            dict_list[composition][charge] = {}

        if spin not in dict_list[composition][charge].keys():
            dict_list[composition][charge][spin] = []

        dict_list[composition][charge][spin].append(file)
        test_out = read(file, index=":")
        total_count += len(test_out)
    print("Total number of frames: {}".format(total_count))
    # print the number of items in the dict
    count = 0
    for comp, dict_charge in dict_list.items():
        for charge, dict_temp in dict_charge.items():
            for spin, dict_sub in dict_temp.items():
                count += len(dict_sub)

    print("Number of items in dict: {}".format(count))

    # for each charge, spin, merge the files, put in
    out_folder = options["merge_sparse_out"]
    # check if folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    total_count = 0
    for comp, dict_charge in dict_list.items():
        for charge, dict_temp in dict_charge.items():
            for spin, dict_sub in dict_temp.items():
                file_out = out_folder + "{}_{}_{}_sparse.xyz".format(comp, charge, spin)
                merge_xyz_files(dict_sub, file_out)
                test_out = read(file_out, index=":")
                total_count += len(test_out)

    print("Total number of frames: {}".format(total_count))


main()
