set -x
set -e


rm -rf atom_indices.dat atom_pairs atom_pairs_tica kcenters_clusters \
    mymsm bayesmsm kmeans_centers kmedoids_centers

# workflow 1
msmb AlanineDipeptide
msmb AtomIndices --out atom_indices.dat -p ~/mixtape_data/alanine_dipeptide/ala2.pdb \
     -d --heavy
msmb AtomPairsFeaturizer --out atom_pairs --trjs '~/mixtape_data/alanine_dipeptide/*.dcd' \
    --pair_indices atom_indices.dat --top ~/mixtape_data/alanine_dipeptide/ala2.pdb
msmb tICA --inp atom_pairs/ --transformed atom_pairs_tica/  --n_components 4 --gamma 0 --weighted_transform \
    --lag_time 2
msmb KCenters --inp atom_pairs_tica --transformed kcenters_clusters --metric cityblock
msmb MarkovStateModel --inp kcenters_clusters --out mymsm
msmb BayesianMarkovStateModel --inp kcenters_clusters --out bayesmsm --n_samples 1000

# 1.1
msmb KMeans --inp atom_pairs_tica --transformed kmeans_centers --n_clusters 10
msmb MarkovStateModel --inp kmeans_centers --out mymsm-kmeans

# 1.2
msmb MiniBatchKMedoids --n_clusters 100 --batch_size 1000 --inp atom_pairs_tica --transformed kmedoids_centers
msmb MarkovStateModel --inp kmedoids_centers --out /dev/null

# 1.3
msmb GaussianFusionHMM --inp atom_pairs_tica --out hmm --n_states 8 --n_features 4
