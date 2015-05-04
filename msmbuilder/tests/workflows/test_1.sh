msmb AlanineDipeptide
msmb AtomIndices --out atom_indices.txt -p ~/msmbuilder_data/alanine_dipeptide/ala2.pdb \
     -d --heavy
msmb AtomPairsFeaturizer --transformed atom_pairs --trjs '~/msmbuilder_data/alanine_dipeptide/*.dcd' \
    --pair_indices atom_indices.txt --top ~/msmbuilder_data/alanine_dipeptide/ala2.pdb
msmb tICA -i atom_pairs/ -t atom_pairs_tica.h5  --n_components 4 \
    --gamma 0 \
    --weighted_transform \
    --lag_time 2
msmb KCenters -i atom_pairs_tica.h5 -t kcenters_clusters.h5 --metric cityblock
msmb MarkovStateModel --inp kcenters_clusters.h5 --out mymsm.pkl
msmb ImpliedTimescales -i kcenters_clusters.h5 -l 1:10 --n_jobs 2
cat timescales.csv
