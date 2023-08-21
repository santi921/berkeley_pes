import glob
import json
import os


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
    out_sparse_folder = options["out_sparse_folder"]

    # using glob to get all xyz files in the folder

    print("root folder: {}".format(root_folder))
    files_target = glob.glob(root_folder + "/*/*/*_sparse.xyz")
    # print(files_target)
    print("Number of files:        {}".format(len(files_target)))

    # organize into list of list where each sublist is a folder
    dict_list = {}
    for file in files_target:
        charge = file.split("/")[-3]
        spin = file.split("/")[-2]
        # print("charge: {}, spin: {}".format(charge, spin))
        if charge not in dict_list.keys():
            dict_list[charge] = {}

        if spin not in dict_list[charge].keys():
            dict_list[charge][spin] = []

        dict_list[charge][spin].append(file)

    # print the number of items in the dict
    count = 0
    for charge, dict_temp in dict_list.items():
        for spin, dict_sub in dict_temp.items():
            count += len(dict_sub)

    print("Number of items in dict: {}".format(count))

    # for each charge, spin, merge the files, put in
    out_folder = options["merge_sparse_out"]
    # check if folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for charge, dict_temp in dict_list.items():
        for spin, dict_sub in dict_temp.items():
            file_out = out_folder + "{}_{}_sparse.xyz".format(charge, spin)
            merge_xyz_files(dict_sub, file_out)


main()
