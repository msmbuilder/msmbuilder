msmb AlanineDipeptide --data_home ./
msmb AtomIndices --out atom_indices.txt \
     -p ./alanine_dipeptide/ala2.pdb \
     -d --heavy

msmb AtomPairsFeaturizer --transformed atom_pairs/ \
    --trjs './alanine_dipeptide/*.dcd' \
    --pair_indices atom_indices.txt \
    --top ./alanine_dipeptide/ala2.pdb \
    --out atom_pairs.pkl

msmb MiniBatchKMeans --n_clusters 100 \
    --batch_size 1000 \
    --inp atom_pairs \
    --transformed kmedoids_centers.h5

msmb MarkovStateModel --inp kmedoids_centers.h5 \
    --out msm.pkl

msmb tICA --inp atom_pairs/ --transformed atom_pairs_tica.h5 \
    --n_components 4 \
    --kinetic_mapping \
    --lag_time 2

msmb GaussianHMM --inp atom_pairs_tica.h5 \
    --out hmm.pkl \
    --n_states 2
