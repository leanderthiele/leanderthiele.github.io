// WASM kernel for cosmopower-style emulators.
//
// The .bin files written by tools/export_weights.py are self-describing.
// JS hands us a pointer + length; cp_load parses the header, copies the
// weights out of the blob, and returns an opaque integer handle.
// cp_predict runs the forward pass (input norm -> N matmul/activation
// pairs -> optional PCA reconstruction -> feature denorm) and writes the
// result into a JS-owned float buffer.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emscripten/emscripten.h>

#define MAX_MODELS 8

typedef struct {
    int in_use;
    int n_inputs;
    int n_layers;          // number of weight matrices
    int has_pca;
    int n_outputs;         // final visible output count
    int output_kind;       // 0 = linear, 1 = log10
    int ell_min, ell_max;
    int* layer_dims;       // n_layers + 1 entries

    float*  input_mean;
    float*  input_std;
    float** W;             // [n_layers] row-major (in_dim x out_dim)
    float** b;             // [n_layers]
    float** alpha;         // [n_layers - 1]
    float** beta;          // [n_layers - 1]

    float* pca_mean;       // [layer_dims[n_layers]]   (only if has_pca)
    float* pca_std;        // [layer_dims[n_layers]]
    float* pca_basis;      // row-major (N_PCA x n_outputs)

    float* features_mean;  // [n_outputs]
    float* features_std;   // [n_outputs]

    float* scratch_a;
    float* scratch_b;
} Model;

static Model g_models[MAX_MODELS];

EMSCRIPTEN_KEEPALIVE int  cp_malloc(int n_bytes) { return (int)(uintptr_t)malloc(n_bytes); }
EMSCRIPTEN_KEEPALIVE void cp_mfree (int ptr)     { free((void*)(uintptr_t)ptr); }

static uint32_t rd_u32(const uint8_t* p, int off) {
    return (uint32_t)p[off]
         | ((uint32_t)p[off + 1] << 8)
         | ((uint32_t)p[off + 2] << 16)
         | ((uint32_t)p[off + 3] << 24);
}

static float* dup_floats(const uint8_t* blob, int* off, int n) {
    float* out = (float*)malloc((size_t)n * sizeof(float));
    memcpy(out, blob + *off, (size_t)n * sizeof(float));
    *off += n * (int)sizeof(float);
    return out;
}

EMSCRIPTEN_KEEPALIVE void cp_free(int handle) {
    if (handle < 0 || handle >= MAX_MODELS) return;
    Model* m = &g_models[handle];
    if (!m->in_use) return;
    int nL = m->n_layers;
    free(m->layer_dims);
    free(m->input_mean); free(m->input_std);
    if (m->W) { for (int i = 0; i < nL; i++) free(m->W[i]); free(m->W); }
    if (m->b) { for (int i = 0; i < nL; i++) free(m->b[i]); free(m->b); }
    if (m->alpha) { for (int i = 0; i < nL - 1; i++) free(m->alpha[i]); free(m->alpha); }
    if (m->beta)  { for (int i = 0; i < nL - 1; i++) free(m->beta[i]);  free(m->beta);  }
    free(m->pca_mean); free(m->pca_std); free(m->pca_basis);
    free(m->features_mean); free(m->features_std);
    free(m->scratch_a); free(m->scratch_b);
    memset(m, 0, sizeof(Model));
}

EMSCRIPTEN_KEEPALIVE int cp_load(int blob_ptr, int blob_len) {
    const uint8_t* blob = (const uint8_t*)(uintptr_t)blob_ptr;
    if (blob_len < 36) return -1;
    if (blob[0] != 'C' || blob[1] != 'P' || blob[2] != 'N' || blob[3] != 'N') return -1;
    if (rd_u32(blob, 4) != 1) return -1;

    int handle = -1;
    for (int i = 0; i < MAX_MODELS; i++) {
        if (!g_models[i].in_use) { handle = i; break; }
    }
    if (handle < 0) return -1;

    Model* m = &g_models[handle];
    memset(m, 0, sizeof(Model));
    m->in_use = 1;

    int off = 8;
    m->n_inputs    = (int)rd_u32(blob, off); off += 4;
    m->n_layers    = (int)rd_u32(blob, off); off += 4;
    m->has_pca     = (int)rd_u32(blob, off); off += 4;
    m->n_outputs   = (int)rd_u32(blob, off); off += 4;
    m->output_kind = (int)rd_u32(blob, off); off += 4;
    m->ell_min     = (int)rd_u32(blob, off); off += 4;
    m->ell_max     = (int)rd_u32(blob, off); off += 4;

    int nL = m->n_layers;
    m->layer_dims = (int*)malloc((size_t)(nL + 1) * sizeof(int));
    for (int i = 0; i <= nL; i++) {
        m->layer_dims[i] = (int)rd_u32(blob, off); off += 4;
    }

    m->input_mean = dup_floats(blob, &off, m->n_inputs);
    m->input_std  = dup_floats(blob, &off, m->n_inputs);

    m->W     = (float**)calloc((size_t)nL, sizeof(float*));
    m->b     = (float**)calloc((size_t)nL, sizeof(float*));
    m->alpha = (float**)calloc((size_t)(nL > 1 ? nL - 1 : 0), sizeof(float*));
    m->beta  = (float**)calloc((size_t)(nL > 1 ? nL - 1 : 0), sizeof(float*));

    for (int i = 0; i < nL; i++) {
        int in_d  = m->layer_dims[i];
        int out_d = m->layer_dims[i + 1];
        m->W[i] = dup_floats(blob, &off, in_d * out_d);
        m->b[i] = dup_floats(blob, &off, out_d);
        if (i < nL - 1) {
            m->alpha[i] = dup_floats(blob, &off, out_d);
            m->beta[i]  = dup_floats(blob, &off, out_d);
        }
    }

    if (m->has_pca) {
        int n_pca = m->layer_dims[nL];
        m->pca_mean  = dup_floats(blob, &off, n_pca);
        m->pca_std   = dup_floats(blob, &off, n_pca);
        m->pca_basis = dup_floats(blob, &off, n_pca * m->n_outputs);
    }
    m->features_mean = dup_floats(blob, &off, m->n_outputs);
    m->features_std  = dup_floats(blob, &off, m->n_outputs);

    if (off != blob_len) {
        cp_free(handle);
        return -1;
    }

    int max_dim = m->n_outputs;
    for (int i = 0; i <= nL; i++) {
        if (m->layer_dims[i] > max_dim) max_dim = m->layer_dims[i];
    }
    m->scratch_a = (float*)malloc((size_t)max_dim * sizeof(float));
    m->scratch_b = (float*)malloc((size_t)max_dim * sizeof(float));

    return handle;
}

EMSCRIPTEN_KEEPALIVE int cp_n_outputs (int h) { return g_models[h].n_outputs;   }
EMSCRIPTEN_KEEPALIVE int cp_output_kind(int h){ return g_models[h].output_kind; }
EMSCRIPTEN_KEEPALIVE int cp_ell_min   (int h) { return g_models[h].ell_min;     }
EMSCRIPTEN_KEEPALIVE int cp_ell_max   (int h) { return g_models[h].ell_max;     }
EMSCRIPTEN_KEEPALIVE int cp_n_inputs  (int h) { return g_models[h].n_inputs;    }

static inline float sigmoidf(float x) {
    // Numerically stable: avoid expf of large positive args.
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

EMSCRIPTEN_KEEPALIVE void cp_predict(int handle, int params_ptr, int out_ptr) {
    Model* m = &g_models[handle];
    const float* params = (const float*)(uintptr_t)params_ptr;
    float* out          = (float*)(uintptr_t)out_ptr;
    int nL = m->n_layers;

    float* a = m->scratch_a;
    float* w = m->scratch_b;

    for (int i = 0; i < m->n_inputs; i++) {
        a[i] = (params[i] - m->input_mean[i]) / m->input_std[i];
    }

    for (int layer = 0; layer < nL; layer++) {
        int in_d  = m->layer_dims[layer];
        int out_d = m->layer_dims[layer + 1];
        const float* W = m->W[layer];
        const float* bv = m->b[layer];

        for (int j = 0; j < out_d; j++) w[j] = bv[j];
        for (int k = 0; k < in_d; k++) {
            float ak = a[k];
            const float* Wrow = W + (size_t)k * out_d;
            for (int j = 0; j < out_d; j++) w[j] += Wrow[j] * ak;
        }

        if (layer < nL - 1) {
            const float* al = m->alpha[layer];
            const float* be = m->beta[layer];
            for (int j = 0; j < out_d; j++) {
                float t = w[j];
                float s = sigmoidf(al[j] * t);
                w[j] = (be[j] + (1.0f - be[j]) * s) * t;
            }
        }

        float* tmp = a; a = w; w = tmp;
    }

    if (m->has_pca) {
        int n_pca = m->layer_dims[nL];
        for (int i = 0; i < n_pca; i++) {
            a[i] = a[i] * m->pca_std[i] + m->pca_mean[i];
        }
        for (int j = 0; j < m->n_outputs; j++) w[j] = 0.0f;
        const float* basis = m->pca_basis;
        for (int k = 0; k < n_pca; k++) {
            float ak = a[k];
            const float* row = basis + (size_t)k * m->n_outputs;
            for (int j = 0; j < m->n_outputs; j++) w[j] += row[j] * ak;
        }
        for (int j = 0; j < m->n_outputs; j++) {
            out[j] = w[j] * m->features_std[j] + m->features_mean[j];
        }
    } else {
        for (int j = 0; j < m->n_outputs; j++) {
            out[j] = a[j] * m->features_std[j] + m->features_mean[j];
        }
    }
}

// ===========================================================================
// Spherical-harmonic synthesis (synfast / HEALPix alm2map style)
// ===========================================================================
// Draw a Gaussian random realization of a real scalar field from an angular
// power spectrum C_ell and synthesize it onto an equirectangular (theta,phi)
// grid using the iso-latitude-ring algorithm: one Legendre transform per
// colatitude ring -> F_m(theta) = sum_ell a_lm Pbar_lm(theta), followed by an
// FFT over phi. The JS side reprojects the grid onto a Mollweide image.
//
// Grid layout (row-major, ntheta x nphi):
//   row i  -> colatitude theta_i = pi * (i + 0.5) / ntheta
//   col j  -> longitude  phi_j   = 2*pi * j / nphi
// Requirements: nphi is a power of two and nphi > 2*lmax.
//
// Conventions: Pbar_lm are fully-normalized associated Legendre functions, so
// Y_lm = Pbar_lm(cos theta) e^{i m phi} with int |Y_lm|^2 dOmega = 1, and
// <|a_lm|^2> = C_ell. Monopole/dipole (ell<2) are set to zero.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define SHT_INV_SQRT_4PI 0.28209479177387814  // 1/sqrt(4 pi)
#define SHT_SQRT_HALF    0.70710678118654752   // 1/sqrt(2)
#define SHT_TWO_PI       6.28318530717958648

// --- tiny PRNG (xorshift32) + Box-Muller gaussian --------------------------
static uint32_t g_rng;
static double   g_gauss_spare;
static int      g_gauss_has;

static inline uint32_t xorshift32(void) {
    uint32_t x = g_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng = x;
    return x;
}
static inline double rng_uniform(void) {
    return ((double)xorshift32() + 0.5) * (1.0 / 4294967296.0);
}
static double rng_gauss(void) {
    if (g_gauss_has) { g_gauss_has = 0; return g_gauss_spare; }
    double u1 = rng_uniform(), u2 = rng_uniform();
    double r = sqrt(-2.0 * log(u1)), t = SHT_TWO_PI * u2;
    g_gauss_spare = r * sin(t);
    g_gauss_has = 1;
    return r * cos(t);
}

// --- in-place iterative radix-2 FFT, exp(+i...) (inverse / synthesis) -------
// tw[k] = exp(+i 2*pi*k/n) for k in [0, n/2).
static void fft_pow2(double* re, double* im, int n,
                     const double* tw_re, const double* tw_im) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len) {
            int k = 0;
            for (int a = i; a < i + half; a++) {
                int b = a + half;
                double wr = tw_re[k], wi = tw_im[k];
                double xr = re[b] * wr - im[b] * wi;
                double xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
                k += step;
            }
        }
    }
}

// cl_ptr : float[lmax+1], C_ell in physical units^2 (ell<2 ignored)
// grid_ptr: float[ntheta*nphi] output, row-major
EMSCRIPTEN_KEEPALIVE
void sht_synth(int cl_ptr, int lmax, int ntheta, int nphi, int seed, int grid_ptr) {
    const float* cl  = (const float*)(uintptr_t)cl_ptr;
    float*       grid = (float*)(uintptr_t)grid_ptr;
    int L = lmax;

    g_rng = seed ? (uint32_t)seed : 0x9e3779b9u;
    g_gauss_has = 0;

    size_t ntri = (size_t)(L + 1) * (L + 2) / 2;

    // triangular offset of block m (entries for ell = m..L)
    int* off = (int*)malloc((size_t)(L + 1) * sizeof(int));
    for (int m = 0, acc = 0; m <= L; m++) { off[m] = acc; acc += (L - m + 1); }

    // Legendre ell-recurrence coefficients (theta-independent, computed once):
    //   Pbar_lm = A_lm * x * Pbar_{l-1,m} - B_lm * Pbar_{l-2,m}   (l >= m+2)
    double* A = (double*)malloc(ntri * sizeof(double));
    double* B = (double*)malloc(ntri * sizeof(double));
    for (int m = 0; m <= L; m++) {
        int o = off[m];
        for (int l = m; l <= L; l++) {
            int t = l - m;
            if (l >= m + 2) {
                A[o + t] = sqrt(((2.0*l + 1.0) * (2.0*l - 1.0)) /
                                ((double)(l - m) * (l + m)));
                B[o + t] = sqrt(((2.0*l + 1.0) * (l + m - 1.0) * (l - m - 1.0)) /
                                ((2.0*l - 3.0) * (l - m) * (l + m)));
            } else { A[o + t] = 0.0; B[o + t] = 0.0; }
        }
    }

    // a_lm (m >= 0; reality fixes m < 0). m=0 real; m>0 complex, var split.
    double* ar = (double*)calloc(ntri, sizeof(double));
    double* ai = (double*)calloc(ntri, sizeof(double));
    for (int l = 2; l <= L; l++) {
        double s = (cl[l] > 0.0f) ? sqrt((double)cl[l]) : 0.0;
        double inv = s * SHT_SQRT_HALF;
        ar[off[0] + l] = s * rng_gauss();           // a_l0 real
        for (int m = 1; m <= l; m++) {
            ar[off[m] + (l - m)] = inv * rng_gauss();
            ai[off[m] + (l - m)] = inv * rng_gauss();
        }
    }

    // twiddles for the phi FFT
    int nh = nphi >> 1;
    double* twr = (double*)malloc((size_t)nh * sizeof(double));
    double* twi = (double*)malloc((size_t)nh * sizeof(double));
    for (int k = 0; k < nh; k++) {
        double ang = SHT_TWO_PI * k / nphi;
        twr[k] = cos(ang); twi[k] = sin(ang);
    }

    double* Fr = (double*)malloc((size_t)(L + 1) * sizeof(double));
    double* Fi = (double*)malloc((size_t)(L + 1) * sizeof(double));
    double* re = (double*)malloc((size_t)nphi * sizeof(double));
    double* im = (double*)malloc((size_t)nphi * sizeof(double));

    for (int i = 0; i < ntheta; i++) {
        double theta = M_PI * (i + 0.5) / ntheta;
        double x = cos(theta), sth = sin(theta);

        // per-ring Legendre transform: F_m = sum_{l>=m} a_lm Pbar_lm(theta)
        double pmm = SHT_INV_SQRT_4PI;   // Pbar_00
        for (int m = 0; m <= L; m++) {
            if (m > 0) pmm *= sqrt((2.0*m + 1.0) / (2.0*m)) * sth;  // Pbar_mm
            const double* Ar = ar + off[m]; const double* Ai = ai + off[m];
            const double* Am = A  + off[m]; const double* Bm = B  + off[m];
            double fr = 0.0, fi = 0.0;
            double p0 = pmm;                       // l = m
            fr += Ar[0] * p0; fi += Ai[0] * p0;
            if (m + 1 <= L) {
                double p1 = sqrt(2.0*m + 3.0) * x * pmm;   // l = m+1
                fr += Ar[1] * p1; fi += Ai[1] * p1;
                for (int l = m + 2; l <= L; l++) {
                    int t = l - m;
                    double p = Am[t] * x * p1 - Bm[t] * p0;
                    fr += Ar[t] * p; fi += Ai[t] * p;
                    p0 = p1; p1 = p;
                }
            }
            Fr[m] = fr; Fi[m] = fi;
        }

        // assemble half-spectrum X[m] = (m?2:1)*F_m, then inverse FFT over phi.
        // The real field is Re(inverse FFT); the Hermitian partner is discarded.
        for (int k = 0; k < nphi; k++) { re[k] = 0.0; im[k] = 0.0; }
        re[0] = Fr[0]; im[0] = Fi[0];
        for (int m = 1; m <= L; m++) { re[m] = 2.0 * Fr[m]; im[m] = 2.0 * Fi[m]; }
        fft_pow2(re, im, nphi, twr, twi);

        float* row = grid + (size_t)i * nphi;
        for (int j = 0; j < nphi; j++) row[j] = (float)re[j];
    }

    free(off); free(A); free(B); free(ar); free(ai);
    free(twr); free(twi); free(Fr); free(Fi); free(re); free(im);
}
