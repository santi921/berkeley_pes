import numpy as np 
import pandas as pd
from tqdm import tqdm

file_rapter = "../../data/20230414_rapter_tracks_initial.json"
#file_libe = "../../data/tasks_opt_trajectories_partial.json"
#df_rapter = dd.read_json(file_rapter)
#df_libe = dd.read_json(file_libe)
df = pd.read_json(file_rapter)
# read first 100 lines 
df = df.head(100)
df.to_json("../../data/test_rapter.json")