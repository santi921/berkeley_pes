import os 
from glob import glob
from berkeley_pes.source.utils.io import write_dict_to_ase_trajectory
from berkeley_pes.source.utils.parse import parse_json

def merge_folders(folder, out_name): 
    # merge all files ending with *xyz in the folder and save them as out_name
    # folder: folder containing the files to merge
    # out_name: name of the merged file
    
    # traverse all files in the folder up to depth 3 
    files = glob(folder + "/**/*.xyz", recursive=True)
    # open out_name file 
    with open(folder + out_name, "w") as outfile:
        for file in files:
            #print(file)
            # open each file and append it to the outfile
            with open(file) as infile:
                for line in infile:
                    outfile.write(line)

def main():
    #rapter_file = (
    #    "/home/santiagovargas/dev/berkeley_pes/data/test_rapter.json"
    #)

    rapter_file = (
        "/home/santiagovargas/dev/berkeley_pes/data/20230414_rapter_tracks_initial.json"
    )
    data_rapter = parse_json(rapter_file, mode="normal", verbose=True)
    print("number of frames: {}".format(len(data_rapter)))
    root_out = "../../../data/ase/rapter_full_splits/"
    write_dict_to_ase_trajectory(
        data_rapter,
        root_out,
        separate_composition=True,
        separate_spin=True,
        separate_charges=True,
        train_prop=0.7,
        val_prop=0.15,
        test_prop=0.15,
    )
    
    val_folder = root_out + "val/"
    train_folder = root_out + "train/"
    test_folder = root_out + "test/"
    
    merge_folders(val_folder, "val.xyz")
    merge_folders(train_folder, "train.xyz")
    merge_folders(test_folder, "test.xyz")



main()
