msmb AlanineDipeptide
msmb AtomIndices --out atom_indices.dat -p ~/mixtape_data/alanine_dipeptide/ala2.pdb \
     -d --heavy
msmb AtomPairsFeaturizer --out atom_pairs --trjs '~/mixtape_data/alanine_dipeptide/*.dcd' \
    --pair_indices atom_indices.dat --top ~/mixtape_data/alanine_dipeptide/ala2.pdb
msmb tICA --inp atom_pairs/ --transformed atom_pairs_tica/  --n_components 4 --gamma 0 --weighted_transform \
    --lag_time 2
msmb KCenters --inp atom_pairs_tica --transformed kcenters_clusters --metric cityblock
msmb MarkovStateModel --inp kcenters_clusters --out mymsm
