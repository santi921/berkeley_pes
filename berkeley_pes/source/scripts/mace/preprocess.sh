python /home/santiagovargas/dev/mace/scripts/preprocess_data.py \
    --train_file="/home/santiagovargas/dev/berkeley_pes/data/ase/20230414_rapter_tracks_initial.xyz" \
    --atomic_numbers="[1, 3, 6, 7, 8, 9, 12, 15, 16, 17]"\
    --h5_prefix="/home/santiagovargas/dev/berkeley_pes/data/ase/processed_data_rapter_v1/" \
    --compute_statistics \
    --E0s="average" \
    --seed=123 \
    --valid_fraction=0.05 
