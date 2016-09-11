msmb AlanineDipeptide --data_home ./
msmb AtomIndices --out atom_indices.txt -p ./alanine_dipeptide/ala2.pdb -d --heavy
msmb AtomPairsFeaturizer --transformed atom_pairs --trjs './alanine_dipeptide/*.dcd' \
    --pair_indices atom_indices.txt --top ./alanine_dipeptide/ala2.pdb --out atom_pairs.pkl
msmb RobustScaler -i atom_pairs/ -t scaled_atom_pairs.h5
msmb tICA -i scaled_atom_pairs.h5 -t atom_pairs_tica.h5  --n_components 4 \
    --shrinkage 0 \
    --kinetic_mapping \
    --lag_time 2
msmb KCenters -i atom_pairs_tica.h5 -t kcenters_clusters.h5 --metric cityblock
msmb MarkovStateModel --inp kcenters_clusters.h5 --out mymsm.pkl
