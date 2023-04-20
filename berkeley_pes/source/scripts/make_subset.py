import numpy as np 
import pandas as pd
from tqdm import tqdm
    
#file_rapter = "../../../data/20230414_rapter_tracks_initial.json"
file_libe = "../../../data/tasks_opt_trajectories_partial.json"
df = pd.read_json(file_libe)
df = df.head(100)
df.to_json("../../../data/test_libe.json")