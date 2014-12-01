msmb AlanineDipeptide
msmb AtomIndices --out atom_indices.dat \
     -p ~/mixtape_data/alanine_dipeptide/ala2.pdb \
     -d --heavy
msmb AtomPairsFeaturizer --out atom_pairs \
    --trjs '~/mixtape_data/alanine_dipeptide/*.dcd' \
    --pair_indices atom_indices.dat \
    --top ~/mixtape_data/alanine_dipeptide/ala2.pdb
msmb MiniBatchKMeans --n_clusters 100 \
    --batch_size 1000 \
    --inp atom_pairs \
    --transformed kmedoids_centers
msmb MarkovStateModel --inp kmedoids_centers \
    --out msm.pkl

msmb tICA --inp atom_pairs/ --transformed atom_pairs_tica/ \
    --n_components 4 \
    --gamma 0 \
    --weighted_transform \
    --lag_time 2
msmb GaussianFusionHMM --inp atom_pairs_tica \
    --out hmm \
    --n_states 2 \
    --n_features 4
