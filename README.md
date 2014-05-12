## mixtape: Statistical models for Biomolecular Dynamics

Mixtape is a python package which implements a series of statistical models for high-dimensional time-series -- the particular focus of the library is on the  analysis of atomistic simulations of biomolecular dynamics such as protein folding and conformational change. Mixtape is available under the LGPL (v2.1 or later).

Mixtape is under active development. Many of the algorithms and models implemented are "mature", but others are rapidly changing.

Algorithms available include:

- Markov state models [1, 2, and references therein].
- Time-structure independent components analysis [3, 4, 5].
- L1-regularized reversible hidden Markov models [6].


###Implementation
Mixtape is a python package. A limited command-line interface is available.
Parts of the code is implemented in python, cython, c/c++, and cuda.

###Dependeices and Installation

```
python setup.py install
```

Python dependencies include: `numpy`, `scipy`, `sklearn`, `mdtraj`, `cython`, `pandas`,  Some modules have further dependencies, including `msmbuilder`, `IPython`, and `cvxopt`. Obviously python is also required (2.6, 2.7, or 3.2+)

### References
1. [Prinz, Jan-Hendrik, et al. J. Chem. Phys. 134.17 (2011): 174105.](http://dx.doi.org/10.1063/1.3565032)
2. [Pande, Vijay S., Kyle Beauchamp, and Gregory R. Bowman. Methods 52.1 (2010): 99-105.](http://dx.doi.org/10.1016/j.ymeth.2010.06.002)
3. [Schwantes, Christian R., and Vijay S. Pande. J. Chem Theory Comput. 9.4 (2013): 2000-2009.](http://dx.doi.org/10.1021/ct300878a)
4. [Perez-Hernandez, Guillermo, et al. J Chem. Phys (2013): 015102.](http://dx.doi.org/10.1063/1.4811489)
5. [Naritomi, Yusuke, and Sotaro Fuchigami. J. Chem. Phys. 134.6 (2011): 065101.](http://dx.doi.org/10.1063/1.3554380)
6. [McGibbon, Robert T. et al., Proc. 31st Intl. Conf. on Machine Learning (ICML). 2014.](http://arxiv.org/abs/1405.1444)

