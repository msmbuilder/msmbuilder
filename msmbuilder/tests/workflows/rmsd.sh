msmb AlanineDipeptide --data_home ./
msmb AtomIndices --out atom_indices.txt \
     -p ./alanine_dipeptide/ala2.pdb \
     -d --heavy

msmb MiniBatchKMedoids --n_clusters 10 \
    --metric rmsd \
    --inp './alanine_dipeptide/*.dcd' \
    --top ./alanine_dipeptide/ala2.pdb \
    --atom_indices atom_indices.txt \
    --transformed kmedoids_centers.h5

msmb RegularSpatial --inp  './alanine_dipeptide/*.dcd' \
    --transformed rs_rmsd.h5 \
    --metric rmsd \
    --top ./alanine_dipeptide/ala2.pdb \
    --d_min 0.5

