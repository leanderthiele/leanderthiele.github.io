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
