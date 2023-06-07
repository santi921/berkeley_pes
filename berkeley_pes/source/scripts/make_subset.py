import pandas as pd
import argparse    

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--file", type=str, help="path to the file to be read", default="../../../data/20230503_jaguar_trajectories.json")
    parser.add_argument("--file", type=str, help="path to the file to be read", default="../../../data/20230414_rapter_tracks_initial.json")
    file = parser.parse_args().file
    print(file)
    df = pd.read_json(file)
    #file_libe = "../../../data/tasks_opt_trajectories_partial.json"
    #df = pd.read_json(file_libe)
    df = df.head(100)
    df.to_json("../../../data/test_subset_rapter.json")

main()