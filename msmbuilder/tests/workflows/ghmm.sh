msmb AlanineDipeptide --data_home ./

msmb DihedralFeaturizer --transformed feats/ \
    --trjs './alanine_dipeptide/*.dcd' \
    --top ./alanine_dipeptide/ala2.pdb \
    --out featy.pkl

msmb tICA --inp feats/ --transformed tica_trajs.h5 \
    --n_components 4 \
    --kinetic_mapping \
    --lag_time 2

msmb GaussianHMM --inp tica_trajs.h5 \
    --out hmm.pkl \
    --n_states 2
