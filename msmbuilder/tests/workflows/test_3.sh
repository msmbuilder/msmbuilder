msmb AlanineDipeptide --data_home ./
msmb AtomIndices --out atom_indices.txt \
     -p ./alanine_dipeptide/ala2.pdb \
     -d --heavy

msmb KCenters --inp  './alanine_dipeptide/*.dcd' \
    --transformed kcenters_rmsd.h5 \
    --metric rmsd \
    --top ./alanine_dipeptide/ala2.pdb \
    --n_clusters 100 \
    --atom_indices atom_indices.txt

msmb RegularSpatial --inp  './alanine_dipeptide/*.dcd' \
    --transformed rs_rmsd.h5 \
    --metric rmsd \
    --top ./alanine_dipeptide/ala2.pdb \
    --d_min 0.1

# h5ls kcenters_rmsd.h5
# h5ls rs_rmsd.h5
