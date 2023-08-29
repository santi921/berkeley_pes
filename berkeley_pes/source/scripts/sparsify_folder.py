import json
import numpy as np

from asaplib.data import ASAPXYZ
from asaplib.reducedim import Dimension_Reducers
from asaplib.compressor import Sparsifier
from asaplib.hypers import (
    gen_default_soap_hyperparameters,
    gen_default_acsf_hyperparameters,
)


def write_subsample_xyz(xyz_file, output_file, frame_inds):
    in_track = -1
    write_trigger = False
    # sort frame_inds
    frame_inds = sorted(frame_inds)

    with open(xyz_file, "r") as f:
        with open(output_file, "w") as g:
            for line in f:
                # check if first character is a digit
                if line[0].isdigit():
                    in_track += 1
                    if in_track in frame_inds:
                        write_trigger = True
                        # write the number of atoms
                        g.write(line)
                    else:
                        write_trigger = False

                else:
                    if write_trigger:
                        g.write(line)


def asap_sparsify(options):
    asapxyz = ASAPXYZ(options["xyz_file"], periodic=options["periodic"])
    n_frames = asapxyz.get_num_frames()
    assert options["min_sparse"] > options["n_components"], "n_components too large"

    if n_frames > options["min_sparse"]:
        energies = asapxyz.get_property("free_energy", sbs=[i for i in range(n_frames)])
        Zs = asapxyz.global_species
        options["Zs"] = Zs
        # descriptor options - atomic
        assert options["descriptor"] in ["SOAP", "ACSF"], "descriptor not supported"
        if options["descriptor"] == "SOAP":
            atomic_spec = gen_default_soap_hyperparameters(
                Zs=options["Zs"],
                soap_n=options["n"],
                soap_l=options["l"],
                multisoap=options["multisoap"],
                sharpness=options["sharpness"],
                scalerange=options["scalerange"],
                verbose=False,
            )
        else:
            atomic_spec = gen_default_acsf_hyperparameters(
                Zs=options["Zs"],
                sharpness=options["sharpness"],
                scalerange=options["scalerange"],
                verbose=False,
            )
        # reducer options
        reducer_spec = {
            "reducer1": {
                "reducer_type": "average",  # [average], [sum], [moment_average], [moment_sum]
                "element_wise": False,
            }
        }

        # descriptor options - global
        desc_spec = {
            "avg_atom": {
                "atomic_descriptor": atomic_spec,
                "reducer_function": reducer_spec,
            }
        }

        # compute atomic descriptors
        asapxyz.compute_atomic_descriptors(
            desc_spec_dict=atomic_spec,
            sbs=[],
            tag="atomic",
            n_process=options["n_process"],
        )
        asapxyz.compute_global_descriptors(
            desc_spec_dict=desc_spec,
            sbs=[],
            keep_atomic=True,  #
            tag="global",
            n_process=options["n_process"],
        )

        # dim reduction options - if you want to move to pca or sort before sparsify
        reduce_dict = {}
        assert options["dim_reduction"] in [
            "PCA",
            "SPARSE_KPCA",
        ], "dim reduction not supported"
        if options["dim_reduction"] == "PCA":
            reduce_dict["pca"] = {
                "type": "PCA",
                "parameter": {
                    "n_components": options["n_components"],
                    "scalecenter": options["scalecenter"],
                },
            }
        else:
            reduce_dict["kpca"] = {
                "type": "SPARSE_KPCA",
                "parameter": {
                    "n_components": options["n_components"],
                    "n_sparse": -1,  # no sparsification
                },
            }
            if options["kernel"] == "linear":
                reduce_dict["kpca"]["parameter"]["kernel"] = {
                    "first_kernel": {"type": "linear"}
                }
            else:
                reduce_dict["kpca"]["parameter"]["kernel"] = {
                    "first_kernel": {"type": "polynomial", "d": 3}
                }

        dreducer = Dimension_Reducers(reduce_dict)
        design_mat = asapxyz.fetch_computed_descriptors(["avg_atom"])
        proj = dreducer.fit_transform(design_mat)

        # mode options
        modes = ["fps", "cur", "random", "sequential"]
        assert options["sparse_mode"] in modes, "sparse mode not supported"

        print("... Now sparsifying ...")
        sparser = Sparsifier(sparse_mode=options["sparse_mode"])
        # get number of structure originally
        n_structures = proj.shape[0]
        if options["n_sparse"] < 1:
            options["n_sparse"] = int(n_structures * options["n_sparse"])
        sparse_inds = sparser.sparsify(
            proj[:], n_or_ratio=options["n_sparse"], sparse_param=1
        )

        if bool(options["save_sparse_inds"]):
            print("... Saving sparse indices to {}".format(options["sparse_inds_file"]))
            np.savetxt(options["sparse_inds_file"], sparse_inds, fmt="%d")

        write_subsample_xyz(
            options["xyz_file"], options["sparse_xyz_file"], sparse_inds
        )
    else:
        print("... No sparsification ...")
        write_subsample_xyz(options["xyz_file"], options["sparse_xyz_file"], [0])


def main():
    # read json with options
    options = json.load(open("./options.json"))
    # read xyz file
    root_folder = options["root_folder"]
    # using glob to get all xyz files in the folder
    import glob

    print("root folder: {}".format(root_folder))
    files_target = glob.glob(root_folder + "/*/*/*.xyz")
    print("Number of files: {}".format(len(files_target)))
    print("First file: {}".format(files_target[0]))

    for file in files_target:
        options_temp = options.copy()
        options_temp["xyz_file"] = file
        options_temp["sparse_xyz_file"] = file.replace(".xyz", "_sparse.xyz")
        options_temp["sparse_inds_file"] = file.replace(".xyz", "_sparse_inds.txt")
        asap_sparsify(options_temp)


main()
