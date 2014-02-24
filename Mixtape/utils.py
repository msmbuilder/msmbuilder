import json
import numpy as np
import sklearn.hmm


def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass

def assign(featurizer, traj, means, covars):
    features = featurizer.featurize(traj)
    
    hmm = sklearn.hmm.GaussianHMM(len(means))
    hmm.means_ = means
    hmm.covars_ = covars
    
    ass = hmm.predict(features)
    probs = hmm.predict_proba(features)

    return ass, probs
