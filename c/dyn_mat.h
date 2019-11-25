#ifndef __dyn_mat_H__
#define __dyn_mat_H__

void calculate_dyn_mat_at_q(const double *qpt, const int n_ions,
    const int n_cells, const int max_ims, const int *n_sc_images,
    const int *sc_image_i, const int *cell_origins, const int *sc_origins,
    const double *fc_mat, double *dyn_mat);

void enforce_reciprocal_asr(const int *ac_i, const double *g_evals,
    const double *g_evecs, const int n_ions, double *dyn_mat);

void mass_weight_dyn_mat(const double* dyn_mat_weighting, const int n_ions,
    double* dyn_mat);

int diagonalise_dyn_mat_zheevd(const int n_ions, double* dyn_mat,
    double* eigenvalues,
    void (*zheevdptr) (char*, char*, int*, double*, int*, double*, double*,
    int*, double*, int*, int*, int*, int*));

void evals_to_freqs(const int n_ions, double *eigenvalues);

#endif