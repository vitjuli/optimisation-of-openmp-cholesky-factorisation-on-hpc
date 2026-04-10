/**
 * @file mphil_dis_cholesky.h
 * @brief Public API for the MPhil DIS C2 Cholesky factorisation library.
 *
 * This header declares the public entry point @ref mphil_dis_cholesky, which
 * computes the in-place Cholesky factorisation \f$C = L\,L^\top\f$ of a
 * symmetric positive definite (SPD) matrix stored in row-major order.
 *
 * @section cho_build_variants Build variants (CHOLESKY_VERSION)
 * The library is compiled in one of four variants selected at build time via
 * -DCHOLESKY_VERSION=N:
 * - 0: Serial baseline (direct transcription of the coursework pseudocode).
 * - 1: Serial optimised (loop interchange, hoisting, reciprocal, unrolling).
 * - 2: OpenMP parallel (persistent parallel region, parallel Schur update).
 * - 3: OpenMP tuned (column scaling moved from omp single into `omp for`).
 *
 * Variants 0 and 1 have no OpenMP dependency. Variants 2 and 3 require OpenMP.
 */

#ifndef MPHIL_DIS_CHOLESKY_H
#define MPHIL_DIS_CHOLESKY_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the in-place Cholesky factorisation \f$C = L\,L^\top\f$.
 *
 * @param[in,out] c Pointer to an \f$n\times n\f$ matrix stored in row-major
 *                  order. The array is overwritten in place.
 * @param[in]     n Matrix dimension.
 *
 * @details
 * Input layout: c[i*n + j] = C(i,j) in row-major order.  \n
 * Output layout (in-place):
 * - Lower triangle (\f$i \ge j\f$): \f$L(i,j)\f$.
 * - Upper triangle (\f$i < j\f$): mirrored values \f$L(j,i)\f$ (i.e.
 * \f$L^\top\f$).
 *
 * After a successful call, the stored result is symmetric
 * (`c[i*n + j] == c[j*n + i]` for \f$i \ne j\f$).
 *
 * Example: for \f$C=\begin{bmatrix}4&2\\2&26\end{bmatrix}\f$ the stored result
 * is \f$\begin{bmatrix}2&1\\1&5\end{bmatrix}\f$.
 *
 * Log-determinant identity: after a successful call,
 * \f[
 * \log|\det C| = 2\sum_{p=0}^{n-1}\log\bigl(c[p*n+p]\bigr).
 * \f]
 *
 * @return
 * - t >= 0.0: wall-clock seconds spent in the factorisation (may be 0.0
 *   for very small problems).
 * - -1.0: invalid input (`c == NULL`, n <= 0, or `n > 100000`).
 * - -2.0: matrix not SPD (a non-positive pivot encountered).
 */
double mphil_dis_cholesky(double *c, int n);

#ifdef __cplusplus
}
#endif

#endif /* MPHIL_DIS_CHOLESKY_H */