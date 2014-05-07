from __future__ import print_function, absolute_import, division
import sys
import six
import time
import numpy as np

from IPython.parallel import Client
from IPython.display import clear_output

try:
    from sklearn.base import BaseEstimator, is_classifier, clone
    from sklearn.metrics.scorer import check_scoring
    from sklearn.utils.validation import _num_samples, check_arrays
    from sklearn.cross_validation import _check_cv as check_cv
    from sklearn.grid_search import (GridSearchCV, BaseSearchCV,
                                     _check_param_grid, ParameterGrid)
    from sklearn.cross_validation import _fit_and_score
except ImportError as e:
    print('This module requires the latest development version (0.15) of sklearn', file=sys.stderr)
    raise e

def _fit_and_score_helper(args):
    from sklearn.cross_validation import _fit_and_score
    return _fit_and_score(*args)


def _wait_interactive(ar, dt=1):
    N = len(ar)
    p = [0 for i in range(N)]

    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        clear_output()
        print("%4i/%i tasks finished after %4i s" % (ar.progress, N, ar.elapsed), end='')
        engine_ids = [md['engine_id'] for md in ar._metadata]
        line_cleared = False
        for i, (eid, stdout) in enumerate(zip(engine_ids, ar.stdout)):
            if eid is None:
                eid = ''
            new_stdout = stdout[p[i]:]
            if new_stdout:
                if not line_cleared:
                    print()
                line_cleared = True
                print("[engine {0}]\n{1}".format(eid, new_stdout))
                p[i] += len(new_stdout)
        sys.stdout.flush()
        time.sleep(dt)

    for i, (eid, stdout) in enumerate(zip(ar.engine_id, ar.stdout)):
        new_stdout = stdout[p[i]:]
        if new_stdout:
            print("[engine {0}]\n{1}".format(eid, new_stdout))
    print()


class DistributedBaseSeachCV(BaseSearchCV):
    def __init__(self, estimator, scoring=None, loss_func=None,
                 score_func=None, fit_params=None, iid=True,
                 refit=True, cv=None, verbose=0, client=None,
                 return_train_scores=True):
        super(DistributedBaseSeachCV, self).__init__(
            estimator=estimator, scoring=scoring, loss_func=loss_func,
            score_func=score_func, iid=iid, refit=refit,
            cv=cv, verbose=verbose)
        self.client = client
        self.return_train_scores = return_train_scores
    
    def _fit(self, X, y, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring,
                                     loss_func=self.loss_func,
                                     score_func=self.score_func)

        n_samples = _num_samples(X)
        X, y = check_arrays(X, y, allow_lists=True, sparse_format='csr',
                            allow_nans=True)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
            y = np.asarray(y)
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        base_estimator = clone(self.estimator)

        client = self.client
        if not isinstance(client, Client):
            client = Client(client)

        view = client.load_balanced_view()
        async = view.map(_fit_and_score_helper,
                       ((clone(base_estimator), X, y, self.scorer_, train, test,
                         self.verbose, parameters, self.fit_params,
                         self.return_train_scores, True)
                for parameters in parameter_iterable
                for train, test in cv), block=False)

        if self.verbose > 0:
            _wait_interactive(async)
        async.wait()
        if self.verbose <= 0:
            async.display_outputs()
        out = async.result

        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            train_scores = []
            all_train_scores = [] if self.return_train_scores else None
            for items in out[grid_start:grid_start + n_folds]:
                # unpack variable number of return values from _fit_and_score
                # depending on self.return_train_scores
                if self.return_train_scores:
                    this_train_score, this_score, this_n_test_samples, \
                        _, parameters = items
                else:
                    this_score, this_n_test_samples, _, parameters = items

                all_scores.append(this_score)
                if self.return_train_scores:
                    all_train_scores.append(this_train_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            # TODO: shall we also store the test_fold_sizes?
            result = {'parameters': parameters,
                      'mean_validation_score': score,
                      'cv_validation_scores': all_scores}
            if self.return_train_scores:
                result['cv_train_scores'] = np.array(all_train_scores)
            grid_scores.append(result)

        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x['mean_validation_score'],
                      reverse=True)[0]
        self.best_params_ = best['parameters']
        self.best_score_ = best['mean_validation_score']

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best['parameters'])
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


class DistributedGridSearchCV(DistributedBaseSeachCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    DistributedGridSearchCV implements a "fit" method and a "predict"
    method like any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    iid : boolean, optional
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : integer or cross-validation generator, optional
        If an integer is passed, it is the number of folds (default 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    refit : boolean
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    client : str, IPython.parallel.Client, optional, default: [use profile]
        IPython.parallel.Client object for distributed map. If not
        supplied, the default client will be constructed. You can
        also path a string, the path to the ipcontroller-client.json file.

    Attributes
    ----------
    `grid_scores_` : list of dicts
        Contains scores for all parameter combinations in param_grid.
        Each entry corresponds to one parameter setting.
        Each dict has the attributes:

            * ``parameters``, a dict of parameter settings
            * ``mean_validation_score``, the mean score over the
              cross-validation folds
            * ``cv_validation_scores``, the list of scores for each fold
            * ``cv_train_scores``, the list of scores computed on the training
              data, if `return_train_scores` is True

    `best_estimator_` : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data.

    `best_score_` : float
        Score of best_estimator on the left out data.

    `best_params_` : dict
        Parameter setting that gave the best results on the hold out data.

    `scorer_` : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a an hyperparameter grid.

    :func:`sklearn.cross_validation.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """

    def __init__(self, estimator, param_grid, scoring=None, loss_func=None,
                 score_func=None, fit_params=None, iid=True,
                 refit=True, cv=None, verbose=0, client=None,
                 return_train_scores=True):
        super(DistributedGridSearchCV, self).__init__(
            estimator, scoring=scoring, loss_func=loss_func,
            score_func=score_func, fit_params=fit_params, iid=iid,
            refit=refit, cv=cv, verbose=verbose, client=client,
            return_train_scores=return_train_scores)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def fit(self, X, y=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        return self._fit(X, y, ParameterGrid(self.param_grid))
