from dask import dataframe as dd
from dask.dataframe.utils import make_meta

# importing dataframes
print("importing dataframes")
file_rapter = "../../data/20230414_rapter_tracks_initial.json"
file_libe = "../../data/tasks_opt_trajectories_partial.json"
df_rapter = dd.read_json(file_rapter)
# df_libe = dd.read_json(file_libe)

# get number of rows
print("getting number of rows")
n_rows = df_rapter.shape[0].compute()


def traj_to_dict(df_row):
    """
    Converts dataframe info into a single dictionary to write
    Takes:
        df_row: a single row of the dataframe
    """
    print(df_row)
    energies = df_row["energy_trajectory"]
    gradients = df_row["gradient_trajectory"]
    # sites = df_row["molecule_trajectory"]
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


res = df_rapter.apply(axis=1, func=traj_to_dict, meta=("object"))
df_out = res.compute()
print(df_out)
print(df_out.shape)
