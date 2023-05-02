import json
import numpy as np
import asaplib
from asaplib.data import ASAPXYZ
from asaplib.plot import Plotters
from asaplib.reducedim import Dimension_Reducers
from asaplib.compressor import Sparsifier


def subsample_xyz(xyz_file, output_file,  frame_inds):
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

def main():
    # read json with options
    options = json.load(open('./options.json'))
    # read xyz file
    asapxyz = ASAPXYZ(options['xyz_file'], periodic = options['periodic'])
    n_frames = asapxyz.get_num_frames()
    energies = asapxyz.get_property("free_energy", sbs=[i for i in range(n_frames)]) 

    # descriptor options - atomic 
    soap_spec = {
        "soap1": {
            "type": "SOAP",
            "cutoff": 10.0,
            "n": 6,
            "l": 6,
            "atom_gaussian_width": 0.5,
            "crossover": False,
            "rbf": "gto",
        }
    }
    # reducer options 
    reducer_spec = {
        "reducer1": {
            "reducer_type": "average",  # [average], [sum], [moment_average], [moment_sum]
            "element_wise": False,
        }
    }

    # descriptor options - global
    desc_spec = {
        "avgsoap": {
            "atomic_descriptor": soap_spec, 
            "reducer_function": reducer_spec}
    }

    # dim reduction options - if you want to move to pca or sort before sparsify
    reduce_dict = {}
    reduce_dict["kpca"] = {
        "type": "SPARSE_KPCA",
        "parameter": {
            "n_components": 10,
            "n_sparse": -1,  # no sparsification
            "kernel": {"first_kernel": {"type": "linear"}},
        },
    }
    # compute atomic descriptors
    asapxyz.compute_atomic_descriptors(desc_spec_dict=soap_spec, sbs=[], tag="atomic", n_process=10)
    asapxyz.compute_global_descriptors(
        desc_spec_dict=desc_spec,
        sbs=[],
        keep_atomic=True,  # set to True to keep the atomic descriptors
        tag="global",
        n_process=10,
    )
    

    dreducer = Dimension_Reducers(reduce_dict)
    design_mat = asapxyz.fetch_computed_descriptors(["avgsoap"])
    proj = dreducer.fit_transform(design_mat)
    
    if bool(options["show"]):
        fig_spec = {
        "outfile": None,
        "show": True,
        "title": None,
        "size": [8 * 1.1, 8],
        "cmap": "gnuplot",
        "xlabel": "PC1",
        "ylabel": "PC2",
        "components": {
            "first_p": {
                "type": "scatter",
                "clabel": "Energies (eV)",
                "vmin": None,
                "vmax": None,
                }
            },
        }
        asap_plot = Plotters(fig_spec)
        plotcolor = energies

        asap_plot.plot(
            proj[[i for i in range(1000)], [1, 0]], 
            plotcolor[[i for i in range(1000)]]
        )

    # mode options
    modes = ["fps", "cur", "random", "sequential"]
    assert options["sparse_mode"] in modes, "sparse mode not supported"
    
    print("... Now sparsifying ...")
    sparser = Sparsifier(sparse_mode=options["sparse_mode"])
    sparse_inds = sparser.sparsify(
        proj[:], 
        n_or_ratio=options["n_sparse"], 
        sparse_param=1
    )
     
    
    if bool(options["save_sparse_inds"]):
        print("... Saving sparse indices to {}".format(options["sparse_inds_file"]))
        np.savetxt(options["sparse_inds_file"], sparse_inds, fmt="%d")
    
    if bool(options["show"]):
        print("Showing a subset of the data")  
        kde_space = proj[sparse_inds]
        kde_space = kde_space[:, [1, 0]]
        plotcolor_sub = energies[sparse_inds]
        asap_plot = Plotters(fig_spec)
        asap_plot.plot(kde_space, plotcolor_sub)
        
    print("global descriptor keys: {}".format(asapxyz.global_desc[0].keys()))
    print("Design matrix shape: {}".format(design_mat.shape))
    print("Number of selected frames: {}".format(len(sparse_inds)))
    print("atomic descriptor keys: {}".format(asapxyz.atomic_desc[0].keys()))
    print("writing subsample xyz file to {}".format(options["sparse_xyz_file"]))
    subsample_xyz(options["xyz_file"], options["sparse_xyz_file"], sparse_inds)
main()