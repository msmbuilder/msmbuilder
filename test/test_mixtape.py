import os
from mdtraj.testing import get_fn, eq
from mdtraj.utils import enter_temp_directory
from mixtape.utils import verboseload
from mixtape.cluster import KMeans
from mixtape.tica import tICA

def test_1():
    fn = get_fn('frame0.h5')
    with enter_temp_directory():
        assert os.system('mixtape DRIDFeaturizer --trjs {} --out a.pkl'.format(fn)) == 0
        assert os.system('mixtape DihedralFeaturizer --types phi psi --trjs {} --out b.pkl'.format(fn)) == 0
        
        assert os.system('mixtape tICA --inp a.pkl --out ticamodel.pkl --transformed tics.pkl') == 0
        assert os.system('mixtape KMeans --random_state 0 --n_init 1 --inp b.pkl --out kmeans.pkl --transformed labels.pkl') == 0

        kmeans0 = verboseload('labels.pkl')
        kmeans1 = KMeans(random_state=0, n_init=1).fit_predict(verboseload('b.pkl'))
        tica0 = verboseload('tics.pkl')
        tica1 = tICA().fit_transform(verboseload('a.pkl'))

    eq(kmeans0[0], kmeans1[0])
    eq(tica0[0], tica1[0])


