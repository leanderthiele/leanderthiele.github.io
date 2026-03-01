#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <math.h>

// global settings
static const uint32_t rng_seed = 13742;
static const int32_t
    L = 32, // side length of lattice
    Nequi = 1024, // number of sweeps to equilibriate
    Nmeas = 128, // number of sweeps to measure
    sweeps_per_meas = 8, // how often we sweep between measurements
    Ntemp = 128; // number of temperatures, linearly spaced, to consider
static const double
    mintemp = 1.0, // minimum temperature
    maxtemp = 3.5; // maximum temperature

// memory
static double *temps;
static int32_t *sweep_pattern; // stores random order of sites to check for flip
static int8_t *samples; // shape Ntemp * Nmeas x (LxL)

// RNG state
static uint32_t Q[4096], c = 362436;

// forward declare the RNG, implementation at the end
static void init_rand (uint32_t seed);
static uint32_t rand_cmwc (void);
static void fisher_yates (int32_t *x, uint32_t n);

static inline int32_t periodic (int32_t x)
{
    return (x % L + L) % L;
}

// the only physics function: it computes the change in Hamiltonian that would be induced
// if the spin at site were flipped
static inline int8_t deltaH (int8_t *lattice, int32_t site)
{
    // this is the nearest neighbours pattern
    static const int32_t offsets[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    int32_t ix = site / L, iy = site % L;
    int8_t out = 0;
    for (int ii=0; ii<4; ++ii)
    {
        int32_t jx = periodic(ix + offsets[ii][0]),
                jy = periodic(iy + offsets[ii][1]),
                other_site = jx * L + jy;

        // the individual terms in Hamiltonian are -si*sj
        // If si=+, Hold=-sj, Hnew=+sj -> delta(H) = +2*sj
        // If si=-, Hold=+sj, Hnew=-sj -> delta(H) = -2*sj
        // In summary, delta(H) = 2*si*sj from this pair
        out += lattice[other_site] * lattice[site];
    }

    return 2*out;
}

static inline void flip (int8_t *lattice, int32_t site)
{
    lattice[site] = -lattice[site];
}

static void sweep (int8_t *lattice, double temp)
{
    for (int32_t ii=0; ii<L*L; ++ii)
    {
        int32_t site = sweep_pattern[ii];
        int8_t _deltaH = deltaH(lattice, site);
        if (_deltaH<0)
            flip(lattice, site);
        else
        {
            double logprob = - (double)(_deltaH) / temp;
            double u = (double)(rand_cmwc()) / (double)(UINT32_MAX); // U(0, 1)
            if (log(u) < logprob)
                flip(lattice, site);
        }
    }
}

int main (int argc, char **argv)
{
    // memory allocation
    sweep_pattern = malloc(L * L * sizeof(int32_t));
    samples = malloc(Ntemp * Nmeas * L * L * sizeof(int8_t));
    temps = malloc(Ntemp * sizeof(double));

    init_rand(rng_seed);

    // initialize the sweep pattern with random permutation
    for (int ii=0; ii<L*L; ++ii) sweep_pattern[ii] = ii;
    fisher_yates(sweep_pattern, L*L);

    // initialize the temperatures
    for (int ii=0; ii<Ntemp; ++ii)
        temps[ii] = mintemp + (double)(ii) * (maxtemp-mintemp) / (double)(Ntemp-1);

    // run Metropolis-Hastings, once per temperature
    for (int ii=0; ii<Ntemp; ++ii)
    {
        int8_t *_samples = samples + ii*Nmeas*L*L; // points to samples for this one
        
        // perform random initialization
        for (int jj=0; jj<L*L; ++jj)
            _samples[jj] = 2 * (int32_t)(rand_cmwc() & 1U) - 1;

        // do the equilibration run
        for (int jj=0; jj<Nequi; ++jj)
            sweep(_samples, temps[ii]);

        // do the measurement run
        for (int jj=1; jj<Nmeas; ++jj)
        {
            memcpy(_samples+jj*L*L, _samples+(jj-1)*L*L, L*L*sizeof(int8_t));
            for (int kk=0; kk<sweeps_per_meas; ++kk)
                sweep(_samples+jj*L*L, temps[ii]);
        }

        if (ii) printf("\r");
        printf("%3d / %d done!", ii+1, Ntemp);
        if (ii+1 == Ntemp) printf("\n");
        fflush(stdout);
    }

    // save to disk
    {
        FILE *fp = fopen("temp.txt", "w");
        for (int ii=0; ii<Ntemp; ++ii)
            fprintf(fp, "%.8f\n", temps[ii]);
        fclose(fp);
    }

    {
        FILE *fp = fopen("samples.bin", "wb");
        fwrite(samples, sizeof(int8_t), Ntemp*Nmeas*L*L, fp);
        fclose(fp);
    }

    // clean up
    free(sweep_pattern);
    free(samples);
    free(temps);
}


// implementation of simple RNG, from https://stackoverflow.com/questions/9492581
static void init_rand(uint32_t x)
{
#define PHI 0x9e3779b9
    Q[0] = x;
    Q[1] = x + PHI;
    Q[2] = x + PHI + PHI;

    for (int ii = 3; ii < 4096; ii++)
        Q[ii] = Q[ii - 3] ^ Q[ii - 2] ^ PHI ^ ii;
#undef PHI
}

static uint32_t rand_cmwc(void)
{
    uint64_t t, a = 18782LL;
    static uint32_t ii = 4095;
    uint32_t x, r = 0xfffffffe;
    ii = (ii + 1) & 4095;
    t = a * Q[ii] + c;
    c = (t >> 32);
    x = t + c;
    if (x < c)
    {
        x++;
        c++;
    }
    return (Q[ii] = r - x);
}

// standard random shuffle, from wikipedia
static void fisher_yates (int32_t *x, uint32_t n)
{
    for (uint32_t ii=n-1; ii>0; --ii)
    {
        uint32_t jj = rand_cmwc() % (ii+1);
        int32_t tmp = x[jj];
        x[jj] = x[ii];
        x[ii] = tmp;
    }
}
