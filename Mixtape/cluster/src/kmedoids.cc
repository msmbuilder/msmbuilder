/* The C clustering library.
 * Copyright (C) 2002 Michiel Jan Laurens de Hoon.
 *
 * This library was written at the Laboratory of DNA Information Analysis,
 * Human Genome Center, Institute of Medical Science, University of Tokyo,
 * 4-6-1 Shirokanedai, Minato-ku, Tokyo 108-8639, Japan.
 * Contact: mdehoon 'AT' gsc.riken.jp
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation with or without modifications and for any purpose and
 * without fee is hereby granted, provided that any copyright notices
 * appear in all copies and that both those copyright notices and this
 * permission notice appear in supporting documentation, and that the
 * names of the contributors or copyright holders not be used in
 * advertising or publicity pertaining to distribution of the software
 * without specific prior permission.
 *
 * THE CONTRIBUTORS AND COPYRIGHT HOLDERS OF THIS SOFTWARE DISCLAIM ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT
 * OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOFTWARE.
 *
 */

#include "kmedoids.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cstring>
#include <numpy/arrayobject.h>
#include <numpy/npy_3kcompat.h>

/* ************************************************************************ */

static int randomassign(npy_intp nclusters, npy_intp nelements,
                        npy_intp clusterid[], PyObject* random);

static void getclustermedoids(npy_intp nclusters, npy_intp nelements,
                              double* distance, npy_intp clusterid[],
                              npy_intp centroids[], double errors[]);

#if PY_MAJOR_VERSION >= 3
static int
#else
static void
#endif
initialize_numpy(void) {
    static int is_initialized = 0;
    if (is_initialized == 0) {
        import_array();
        is_initialized = 1;
    }
}

static inline npy_intp ix(npy_intp i, npy_intp j, npy_intp n) {
    if (i == j) {
        fprintf(stderr, "kmedoids.c: error");
        exit(1);
    }
    if (i < j) {
        return n*i - i*(i+1)/2 + j - 1 - i;
    }
    return n*j - j*(j+1)/2 + i - 1 - j;
}

/* ************************************************************************ */


void kmedoids(npy_intp nclusters, npy_intp nelements, double* distmatrix,
              npy_intp npass, npy_intp clusterid[], PyObject* random,
              double* error, npy_intp* ifound)
/*
Purpose
=======

The kmedoids routine performs k-medoids clustering on a given set of elements,
using the distance matrix and the number of clusters passed by the user.
Multiple passes are being made to find the optimal clustering solution, each
time starting from a different initial clustering.


Arguments
=========

nclusters  (input) int
The number of clusters to be found.

nelements  (input) int
The number of elements to be clustered.

distmatrix (input) double array,
Condensed distance matrix. The lower triangular entries of the symmetric
distance matrix. This is the format returned by ``scipy.spatial.distance.pdist()``

npass      (input) int
The number of times clustering is performed. Clustering is performed npass
times, each time starting from a different (random) initial assignment of genes
to clusters. The clustering solution with the lowest within-cluster sum of
distances is chosen.
If npass==0, then the clustering algorithm will be run once, where the initial
assignment of elements to clusters is taken from the clusterid array.

random     (input) numpy.RandomState
Python numpy.RandomState object

clusterid  (output; input) int[nelements]
On input, if npass==0, then clusterid contains the initial clustering assignment
from which the clustering algorithm starts; all numbers in clusterid should be
between zero and nelements-1 inclusive. If npass!=0, clusterid is ignored on
input.
On output, clusterid contains the clustering solution that was found: clusterid
contains the number of the cluster to which each item was assigned. On output,
the number of a cluster is defined as the item number of the centroid of the
cluster.

error      (output) double
The sum of distances to the cluster center of each item in the optimal k-medoids
clustering solution that was found.

ifound     (output) int
If kmedoids is successful: the number of times the optimal clustering solution
was found. The value of ifound is at least 1; its maximum value is npass.
If the user requested more clusters than elements available, ifound is set
to 0. If kmedoids fails due to a memory allocation error, ifound is set to -1.

========================================================================
*/
{
    npy_intp i, j, icluster;
    npy_intp* tclusterid;
    npy_intp* saved;
    npy_intp* centroids;
    double* errors;
    npy_intp ipass = 0;
    int err;

    if (nelements < nclusters) {
        *ifound = 0;
        return;
    } /* More clusters asked for than elements available */

    *ifound = -1;

    /* We save the clustering solution periodically and check if it reappears */
    saved = reinterpret_cast<npy_intp*>(malloc(nelements*sizeof(npy_intp)));
    if (!saved) return;

    centroids = reinterpret_cast<npy_intp*>(malloc(nclusters*sizeof(npy_intp)));
    if (!centroids) {
        free(saved);
        return;
    }

    errors = reinterpret_cast<double*>(malloc(nclusters*sizeof(double)));
    if (!errors) {
        free(saved);
        free(centroids);
        return;
    }

    /* Find out if the user specified an initial clustering */
    if (npass <= 1)
        tclusterid = clusterid;
    else {
        tclusterid = reinterpret_cast<npy_intp*>(malloc(nelements*sizeof(npy_intp)));
        if (!tclusterid) {
            free(saved);
            free(centroids);
            free(errors);
            return;
        }
    }

    *error = DBL_MAX;
    do { /* Start the loop */
        double total = DBL_MAX;
        npy_intp counter = 0;
        npy_intp period = 10;

        if (npass != 0) {
            err = randomassign(nclusters, nelements, tclusterid, random);
            if (err < 0) {
                *ifound = err;
                return;
            }
        }

        while (1) {
            double previous = total;
            total = 0.0;

            if (counter % period == 0) { /* Save the current cluster assignments */
                for (i = 0; i < nelements; i++) saved[i] = tclusterid[i];
                if (period < NPY_MAX_INTP / 2) period *= 2;
            }
            counter++;

            /* Find the center */
            getclustermedoids(nclusters, nelements, distmatrix, tclusterid,
                              centroids, errors);

            for (i = 0; i < nelements; i++)
                /* Find the closest cluster */
            {
                double distance = DBL_MAX;
                for (icluster = 0; icluster < nclusters; icluster++) {
                    double tdistance;
                    j = centroids[icluster];
                    if (i == j) {
                        distance = 0.0;
                        tclusterid[i] = icluster;
                        break;
                    }

                    tdistance = distmatrix[ix(i, j, nelements)];
                    if (tdistance < distance) {
                        distance = tdistance;
                        tclusterid[i] = icluster;
                    }
                }
                total += distance;
            }
            if (total >= previous) break;
            /* total>=previous is FALSE on some machines even if total and previous
             * are bitwise identical. */
            for (i = 0; i < nelements; i++)
                if (saved[i] != tclusterid[i]) break;
            if (i == nelements)
                break; /* Identical solution found; break out of this loop */
        }

        for (i = 0; i < nelements; i++) {
            if (clusterid[i] != centroids[tclusterid[i]]) {
                if (total < *error) {
                    *ifound = 1;
                    *error = total;
                    /* Replace by the centroid in each cluster. */
                    for (j = 0; j < nelements; j++)
                        clusterid[j] = centroids[tclusterid[j]];
                }
                break;
            }
        }
        if (i == nelements) (*ifound)++; /* break statement not encountered */
    } while (++ipass < npass);

    /* Deallocate temporarily used space */
    if (npass > 1) free(tclusterid);

    free(saved);
    free(centroids);
    free(errors);

    return;
}

/* ********************************************************************* */

static void getclustermedoids(npy_intp nclusters, npy_intp nelements,
                              double* distmatrix, npy_intp clusterid[], npy_intp centroids[],
                              double errors[])
/*
Purpose
=======
The getclustermedoids routine calculates the cluster centroids, given to which
cluster each element belongs. The centroid is defined as the element with the
smallest sum of distances to the other elements.
Arguments
=========
nclusters  (input) int
The number of clusters.
nelements  (input) int
The total number of elements.
distmatrix (input) double array
Condensed distance matrix. The lower triangular entries of the symmetric
distance matrix. This is the format returned by ``scipy.spatial.distance.pdist()``
clusterid  (output) int[nelements]
The cluster number to which each element belongs.
centroid   (output) int[nclusters]
The index of the element that functions as the centroid for each cluster.
errors     (output) double[nclusters]
The within-cluster sum of distances between the items and the cluster
centroid.
========================================================================
*/
{
    npy_intp i, j, k;
    for (j = 0; j < nclusters; j++)
        errors[j] = DBL_MAX;
    for (i = 0; i < nelements; i++) {
        double d = 0.0;
        j = clusterid[i];
        for (k = 0; k < nelements; k++) {
            if (i == k || clusterid[k] != j)
                continue;
            d += distmatrix[ix(i, k, nelements)];
            if (d > errors[j]) break;
        }
        if (d < errors[j]) {
            errors[j] = d;
            centroids[j] = i;
        }
    }
}


/* ************************************************************************ */

static int randomassign(npy_intp nclusters, npy_intp nelements,
                        npy_intp clusterid[], PyObject* random)
/*
Purpose
=======
The randomassign routine performs an initial random clustering, needed for
k-means or k-median clustering. Elements (genes or microarrays) are randomly
assigned to clusters. The number of elements in each cluster is chosen
randomly, making sure that each cluster will receive at least one element.
Arguments
=========
nclusters  (input) int
The number of clusters.
nelements  (input) int
The number of elements to be clustered (i.e., the number of genes or microarrays
to be clustered).
clusterid  (output) int[nelements]
The cluster number to which an element was assigned.
============================================================================
*/
{
    npy_intp i, j;
    npy_intp k = 0;
    double p;
    npy_intp n = nelements-nclusters;
    PyObject *binomial = NULL, *shuffle = NULL, *args = NULL, *result = NULL, *arr = NULL;
    initialize_numpy();
    binomial = PyObject_GetAttrString(random, "binomial");
    if (binomial == NULL) {
        PyErr_SetString(PyExc_AttributeError, "binomial");
        return -1;
    }
    shuffle = PyObject_GetAttrString(random, "shuffle");
    if (shuffle == NULL) {
        Py_DECREF(binomial);
        PyErr_SetString(PyExc_AttributeError, "shuffle");
        return -1;
    }

    /* Draw the number of elements in each cluster from a multinomial
     * distribution, reserving ncluster elements to set independently
     * in order to guarantee that none of the clusters are empty.
     */
    for (i = 0; i < nclusters-1; i++) {
        p = 1.0/(nclusters-i);
        /* j = binomial(n, p); */
        args = Py_BuildValue("(dd)", static_cast<double>(n), p);
        result = PyObject_Call(binomial, args, NULL);
        j = PyInt_AsLong(result);
        Py_DECREF(args);
        Py_DECREF(result);

        n -= j;
        j += k+1; /* Assign at least one element to cluster i */
        for ( ; k < j; k++) clusterid[k] = i;
    }
    /* Assign the remaining elements to the last cluster */
    for ( ; k < nelements; k++) clusterid[k] = i;

    /* Create a random permutation of the cluster assignments */
    arr = PyArray_SimpleNewFromData(1, &nelements, NPY_INTP, reinterpret_cast<void*>(clusterid));
    args = Py_BuildValue("(O)", arr);
    result = PyObject_Call(shuffle, args, NULL);
    Py_DECREF(arr);
    Py_DECREF(args);

    Py_DECREF(binomial);
    Py_DECREF(shuffle);
    return 1;
}


std::map<npy_intp, npy_intp> contigify_ids(npy_intp* ids, npy_intp length) {
    npy_intp i = 0;
    npy_intp id, new_id;
    std::map<npy_intp, npy_intp> mapping;

    for (i = 0; i < length; i++) {
        id = ids[i];
        if (mapping.count(id) == 0) {
            new_id = static_cast<npy_intp>(mapping.size());
            mapping[id] = new_id;
        }
        ids[i] = mapping[id];
    }

    return mapping;
}
