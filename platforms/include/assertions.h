#ifndef _MY_ASSERTIONS_H_
#define _MY_ASSERTIONS_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#define ASSERT_TOL(EXPR,VAL,TOL) {                                      \
        double cval; cval = (EXPR);                                     \
        double err; err = cval - (double)(VAL);                         \
        double relerrpc = (cval-(VAL))/(VAL)*100;                       \
        if(fabs(err)>TOL) {                                             \
            fprintf(stderr,"ERROR in line %d: value of '%s' = %f, should be %f, error is %f (%.2f%%)!n" \
                    , __LINE__, #EXPR, cval, VAL,cval-(VAL),relerrpc);  \
            exit(1);                                                    \
        } else{                                                         \
            fprintf(stderr,"    OK, %s = %8.2e with %.2f%% err.\n",#EXPR,VAL,relerrpc); \
        }                                                               \
    }


#endif
