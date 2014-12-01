#ifndef _CEPHES_H_
#define _CEPHES_H_

#include "cephes_names.h"
int mtherr(char *name, int code);
double i0(double x);
double i1(double x);
double zeta(double x, double q);
double psi(double x);
double lgam(double x);
double p1evl(double x, double coef[], int N);
double polevl(double x, double coef[], int N);
double chbevl(double x, double array[], int n);

#endif
