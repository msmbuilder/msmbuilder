set -x
set -e


rm -rf atom_indices.txt atom_pairs atom_pairs_tica.h5 kcenters_clusters.h5 \
    mymsm-kmeans.pkl mymsm.pkl bayesmsm.pkl kmeans_centers.h5 kmedoids_centers.h5 msm.pkl

# workflow 1
msmb AlanineDipeptide
msmb AtomIndices --out atom_indices.txt -p ~/mixtape_data/alanine_dipeptide/ala2.pdb \
     -d --heavy
msmb AtomPairsFeaturizer --out atom_pairs --trjs '~/mixtape_data/alanine_dipeptide/*.dcd' \
    --pair_indices atom_indices.txt --top ~/mixtape_data/alanine_dipeptide/ala2.pdb
msmb tICA --inp atom_pairs/ --transformed atom_pairs_tica.h5  --n_components 4 --gamma 0 --weighted_transform \
    --lag_time 2
msmb KCenters -i atom_pairs_tica.h5 -t kcenters_clusters.h5 --metric cityblock
msmb MarkovStateModel --i kcenters_clusters.h5 --out mymsm.pkl
msmb BayesianMarkovStateModel -i kcenters_clusters.h5 --out bayesmsm.pkl --n_samples 1000

# 1.1
msmb KMeans --inp atom_pairs_tica.h5 -t kmeans_centers.h5 --n_clusters 10
msmb MarkovStateModel -i kmeans_centers.h5 --out mymsm-kmeans.pkl

# 1.2
msmb MiniBatchKMedoids --n_clusters 100 --batch_size 1000 --inp atom_pairs_tica.h5 -t kmedoids_centers.h5
msmb MarkovStateModel -i kmedoids_centers.h5 --out msm.pkl

# 1.3
msmb GaussianFusionHMM --inp atom_pairs_tica.h5 --out hmm --n_states 8 --n_features 4
