from __future__ import print_function, absolute_import, division
import os
import sys
import six
import time
import datetime
import numpy as np
from scipy.stats import sem
from sklearn.externals.joblib import load, dump
from sklearn.externals.joblib.logger import short_format_time

from IPython.parallel import Client
from IPython.parallel.error import RemoteError
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
    import numpy as np
    from sklearn.externals import six
    from sklearn.externals.joblib import load
    from sklearn.cross_validation import _fit_and_score
    args = list(args)
    if isinstance(args[1], six.string_types):
        args[1] = load(args[1], mmap_mode='c')
        if isinstance(args[1], np.memmap):
            args[1] = np.asarray(args[1])
    return _fit_and_score(*args)


def verbose_wait(amr, clientview, return_train_scores):
    N = len(amr)
    pending = set(amr.msg_ids)
    while pending:
        try:
            clientview.wait(pending, 1e-3)
        except parallel.TimeoutError:
            pass

        n_completed = N - len(clientview.outstanding)
        finished = pending.difference(clientview.outstanding)
        pending = pending.difference(finished)

        if len(finished) > 0:
            print()

        for msg_id in finished:
            ar = clientview.get_result(msg_id)
            try:
                for result in ar.result:
                    elapsed, params = result[-2], result[-1]
                    test_score = result[1] if return_train_scores else result[0]
                    left = '[CV engine={}] {}   '.format(ar.engine_id,
                        ', '.join('{}={}'.format(k, v) for k, v in params.items()))
                    right = '  score = {:5f}  {}'.format(test_score, short_format_time(elapsed))
                    print(left + right.rjust(70-len(left), '-'))
            except RemoteError as e:
                e.print_traceback()
                raise
        else:
            left = '\r[Parallel] {0:d}/{1:d}  tasks finished'.format(n_completed, N)
            right = 'elapsed {0}         '.format(short_format_time(amr.elapsed))
            print(left + right.rjust(71-len(left)), end='')
            sys.stdout.flush()
            time.sleep(1 + round(amr.elapsed) - amr.elapsed)

    n_engines = len(set(e['engine_id'] for e in amr._metadata))
    engine_time = sum((e.completed - e.submitted for e in amr._metadata),
                      datetime.timedelta()).total_seconds()

    m1 = 'Elapsed walltime:    {}'.format(short_format_time(amr.elapsed))
    m2 = 'Elapsed engine time: {}'.format(short_format_time(engine_time))
    m3a = 'Parallel speedup:'
    m3b = '{:.3f}'.format(engine_time/ amr.elapsed).rjust(len(m2)-len(m3a))
    m4a = 'Number of engines:'
    m4b = '{}'.format(n_engines).rjust(len(m2)-len(m4a))
    print('\n\nTasks completed')
    print('-'*len(m2))
    print(m1)
    print(m2)
    print(m3a + m3b)
    print(m4a + m4b)
    print('-'*len(m2))



class DistributedBaseSeachCV(BaseSearchCV):
    def __init__(self, estimator, scoring=None, loss_func=None,
                 score_func=None, fit_params=None, iid=True,
                 refit=True, cv=None, verbose=0, client=None,
                 return_train_scores=False, tmp_dir='.'):
        super(DistributedBaseSeachCV, self).__init__(
            estimator=estimator, scoring=scoring, loss_func=loss_func,
            score_func=score_func, iid=iid, refit=refit,
            cv=cv, verbose=verbose)
        self.client = client
        self.return_train_scores = return_train_scores
        self.tmp_dir = tmp_dir

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
        verbose = self.verbose
        client = self.client
        if not isinstance(client, Client):
            client = Client(client)

        if verbose > 0:
            print('Fitting %d folds for each of %d candidates, totalling %d fits on %d engines' % (
                    len(cv), len(parameter_iterable), len(cv)*len(parameter_iterable),
                    len(client)))

        if self.tmp_dir:
            tmpfn = os.path.abspath(os.path.join(self.tmp_dir,
                'temp-{:05d}.pkl'.format(np.random.randint(1e5))))
            if verbose:
                print("Persisting data to {}".format(tmpfn))
            clean_me_up = dump(X, tmpfn)
            # Warm up the data to avoid concurrent disk access in
            # multiple children processes
            try:
                load(tmpfn, mmap_mode='r').max()
            except AttributeError:
                pass
            Xscatter = tmpfn
        else:
            Xscatter = X

        try:
            view = client.load_balanced_view()
            async = view.map(_fit_and_score_helper,
                             ((clone(base_estimator), Xscatter, y, self.scorer_, train, test,
                               self.verbose, parameters, self.fit_params,
                               self.return_train_scores, True)
                    for parameters in parameter_iterable
                    for train, test in cv), block=False, chunksize=1)

            if verbose > 0:
                verbose_wait(async, view, self.return_train_scores)
            async.wait()
            if verbose > 0:
                async.display_outputs()
            try:
                out = async.result
            except RemoteError as e:
                e.print_traceback()
                raise
        finally:
            if self.tmp_dir:
                if verbose:
                    print("Cleaning up {}".format(tmpfn))
                for fn in clean_me_up:
                    os.unlink(fn)

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
                 return_train_scores=False, tmp_dir='.'):
        super(DistributedGridSearchCV, self).__init__(
            estimator, scoring=scoring, loss_func=loss_func,
            score_func=score_func, fit_params=fit_params, iid=iid,
            refit=refit, cv=cv, verbose=verbose, client=client,
            return_train_scores=return_train_scores, tmp_dir=tmp_dir)
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
