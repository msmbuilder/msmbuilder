# vmhmm : `sklearn`-compatible  von Mises hidden Markov model
### Overview
This package implements a hidden Markov model with von Mises emissions.
See [this page](http://scikit-learn.org/stable/modules/hmm.html) for a 
practical description of hidden Markov models.

The von Mises distribution, (also known as the circular normal
distribution or Tikhonov distribution) is a continuous probability
distribution on the circle. For multivariate signals, the emmissions
distribution implemented by this model is a product of univariate
von Mises distributuons -- analogous to the multivariate Gaussian
distribution with a diagonal covariance matrix.


### Installation
vmhmm requires [scikit-learn](http://scikit-learn.org/stable/). It also
requires numpy and scipy, but these are already depdendencies of sklearn
itself. You'll need a C compiler as well. Python 2.6, 2.7, 3.2, and 3.3
are supported.

The packages uses distutils, which is the default way of installing python
modules. The install command is `python setup.py install`.

### Example

```
from vmhmm import VonMisesHMM

model = VonMisesHMM(n_components=3, n_iter=1000)
model.fit([X])
hidden_states = model.predict(X)
```
![](https://raw.github.com/rmcgibbo/vmhmm/master/example/winddirection.py.png)
