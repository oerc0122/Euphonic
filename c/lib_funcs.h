#ifndef __lib_funcs_H__
#define __lib_funcs_H__

typedef void (*ZheevdFunc)(char* jobz, char* uplo, int* n, double* a, int* lda,
    double* w, double* work, int* lwork, double* rwork, int* lrwork,
    int* iwork, int* liwork, int* info);
    
#endif