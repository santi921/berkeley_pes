from berkeley_pes.source.utils.io import write_dict_to_ase_trajectory
from berkeley_pes.source.utils.parse import parse_json


def main():
    rapter_file = (
        "/home/santiagovargas/dev/berkeley_pes/data/20230414_rapter_tracks_initial.json"
    )
    data_rapter = parse_json(rapter_file, mode="normal", verbose=True)
    print("number of frames: {}".format(len(data_rapter)))

    write_dict_to_ase_trajectory(
        data_rapter,
        "../../../data/ase/rapter_full/",
        separate_composition=True,
        separate_spin=True,
        separate_charges=True,
    )


main()
