/* File:     pi_race.cpp
 * Purpose:  Estimate pi with  pi = 4 * sum_{k=0}^{n-1} (-1)^k / (2k+1)
 *           and compare six ways of combining the sum.
 *             version 0: serial baseline
 *             version 1: shared sum, no synchronisation   (DATA RACE)
 *             version 2: critical inside the loop         (correct, slow)
 *             version 3: atomic inside the loop           (correct, slow)
 *             version 4: private partial sum + one critical per thread
 *             version 5: reduction(+: sum)                (correct, fast)
 *
 *           STARTING POINT: every version is still SERIAL.
 *           Add the OpenMP directives yourself where it says TODO.
 *
 * Compile:  g++ -O2 -Wall -Wextra -fopenmp -o ./bin/pi_race ./src/pi_race.cpp
 * Run:      ./bin/pi_race <thread_count> <n> <version>
 *           e.g. ./bin/pi_race 4 100000000 0
 * Based on: Pacheco & Malensek (2022), Sec. 5.5.4 (estimating pi)
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>     // only for omp_get_wtime() until you parallelize

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <thread_count> <n> <version 0-5>\n", argv[0]);
        return 1;
    }
    int  thread_count = (int) strtol(argv[1], NULL, 10);
    long n            = strtol(argv[2], NULL, 10);
    int  version      = (int) strtol(argv[3], NULL, 10);

    if (thread_count <= 0 || n <= 0) {
        fprintf(stderr, "need thread_count > 0 and n > 0\n");
        return 1;
    }

    double sum = 0.0;
    double start = omp_get_wtime();

    switch (version) {
    case 0:                                   /* serial baseline */
        for (long k = 0; k < n; k++) {
            double factor = (k % 2 == 0) ? 1.0 : -1.0;
            sum += factor / (2*k + 1);
        }
        break;

    case 1:                                   /* RACE: shared sum, no synchronisation */
        /* TODO: parallel for with shared sum (no protection) */
        #pragma omp parallel for num_threads(thread_count) default(none) shared(sum, n) 
        for (long k = 0; k < n; k++) {
            double factor = (k % 2 == 0) ? 1.0 : -1.0;
            sum += factor / (2*k + 1);
        }
        break;

    case 2:                                   /* one thread at a time */
        /* TODO: parallel for + critical around the update */
        #pragma omp parallel for num_threads(thread_count) default(none) shared(sum, n) 
        for (long k = 0; k < n; k++) {
            double factor = (k % 2 == 0) ? 1.0 : -1.0;
            #pragma omp critical
            sum += factor / (2*k + 1);
        }
        break;

    case 3:                                   /* hardware atomic update */
        /* TODO: parallel for + atomic on the update */
        #pragma omp parallel for num_threads(thread_count) default(none) shared(sum, n) 
        for (long k = 0; k < n; k++) {
            double factor = (k % 2 == 0) ? 1.0 : -1.0;
            #pragma omp atomic
            sum += factor / (2*k + 1);
        }
        break;

    case 4:                                   /* private partial sums */
        /* TODO: parallel region, private my_sum, omp for,
                 then add my_sum to sum ONCE per thread (critical) */
        #pragma omp parallel num_threads(thread_count) default(none) shared(sum, n) 
        {
            double my_sum = 0.0;
            #pragma omp for
            for (long k = 0; k < n; k++) {
                double factor = (k % 2 == 0) ? 1.0 : -1.0;
                my_sum += factor / (2*k + 1);
            }
            #pragma omp critical
            sum += my_sum; // cant cause issue because the + addtion is associative property
        }
        break;

    case 5:                                   /* let OpenMP do case 4 */
        /* TODO: parallel for + reduction(+: sum) */
        #pragma omp parallel num_threads(thread_count) default(none) shared(n) reduction(+:sum) 
        for (long k = 0; k < n; k++) {
            double factor = (k % 2 == 0) ? 1.0 : -1.0;
            sum += factor / (2*k + 1);
        }
        break;

    default:
        fprintf(stderr, "version must be 0..5\n");
        return 1;
    }

    double elapsed = omp_get_wtime() - start;
    double pi_est  = 4.0 * sum;

    printf("version %d, threads %2d, n = %ld: pi ~ %.15f, |error| = %.3e, "
           "time = %.4f s\n", version, version == 0 ? 1 : thread_count, n,
           pi_est, fabs(pi_est - M_PI), elapsed);

    return 0;
}



// :~/ for v in 0 1 2 3 4 5; do ./bin/pi_race 2 20000000 $v; done
// version 0, threads  1, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 0.0459 s
// version 1, threads  2, n = 20000000: pi ~ 0.000000050000000, |error| = 3.142e+00, time = 0.0121 s
// version 2, threads  2, n = 20000000: pi ~ 3.141592603589816, |error| = 5.000e-08, time = 0.5233 s
// version 3, threads  2, n = 20000000: pi ~ 3.141592603589816, |error| = 5.000e-08, time = 0.6844 s
// version 4, threads  2, n = 20000000: pi ~ 3.141592603589792, |error| = 5.000e-08, time = 0.0103 s
// version 5, threads  2, n = 20000000: pi ~ 6.283185207179634, |error| = 3.142e+00, time = 0.0209 s
// :~/ for v in 0 1 2 3 4 5; do ./bin/pi_race 4 20000000 $v; done
// version 0, threads  1, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 0.0565 s
// version 1, threads  4, n = 20000000: pi ~ 0.000000033333333, |error| = 3.142e+00, time = 0.0064 s
// version 2, threads  4, n = 20000000: pi ~ 3.141592603589818, |error| = 5.000e-08, time = 1.4237 s
// version 3, threads  4, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 1.1091 s
// version 4, threads  4, n = 20000000: pi ~ 3.141592603589780, |error| = 5.000e-08, time = 0.0122 s
// version 5, threads  4, n = 20000000: pi ~ 12.566370414359268, |error| = 9.425e+00, time = 0.0236 s
// :~/ for v in 0 1 2 3 4 5; do ./bin/pi_race 6 20000000 $v; done
// version 0, threads  1, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 0.0382 s
// version 1, threads  6, n = 20000000: pi ~ 3.141592353589731, |error| = 3.000e-07, time = 0.0044 s
// version 2, threads  6, n = 20000000: pi ~ 3.141592603589815, |error| = 5.000e-08, time = 1.5887 s
// version 3, threads  6, n = 20000000: pi ~ 3.141592603589818, |error| = 5.000e-08, time = 1.3761 s
// version 4, threads  6, n = 20000000: pi ~ 3.141592603589671, |error| = 5.000e-08, time = 0.0036 s
// version 5, threads  6, n = 20000000: pi ~ 18.849555621538901, |error| = 1.571e+01, time = 0.0413 s
// :~/ for v in 0 1 2 3 4 5; do ./bin/pi_race 8 20000000 $v; done
// version 0, threads  1, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 0.0627 s
// version 1, threads  8, n = 20000000: pi ~ 0.000000066666667, |error| = 3.142e+00, time = 0.0057 s
// version 2, threads  8, n = 20000000: pi ~ 3.141592603589816, |error| = 5.000e-08, time = 1.9249 s
// version 3, threads  8, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 1.4879 s
// version 4, threads  8, n = 20000000: pi ~ 3.141592603589742, |error| = 5.000e-08, time = 0.0052 s
// version 5, threads  8, n = 20000000: pi ~ 25.132740828718536, |error| = 2.199e+01, time = 0.0414 s
// :~/ for v in 0 1 2 3 4 5; do ./bin/pi_race 10 20000000 $v; done
// version 0, threads  1, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 0.0534 s
// version 1, threads 10, n = 20000000: pi ~ 0.000000250000000, |error| = 3.142e+00, time = 0.0050 s
// version 2, threads 10, n = 20000000: pi ~ 3.141592603589812, |error| = 5.000e-08, time = 2.3352 s
// version 3, threads 10, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 1.8163 s
// version 4, threads 10, n = 20000000: pi ~ 3.141592603589724, |error| = 5.000e-08, time = 0.0043 s
// version 5, threads 10, n = 20000000: pi ~ 31.415926035898170, |error| = 2.827e+01, time = 0.0413 s
// :~/ for v in 0 1 2 3 4 5; do ./bin/pi_race 12 20000000 $v; done
// version 0, threads  1, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 0.0516 s
// version 1, threads 12, n = 20000000: pi ~ -0.000000349999930, |error| = 3.142e+00, time = 0.0044 s
// version 2, threads 12, n = 20000000: pi ~ 3.141592603589819, |error| = 5.000e-08, time = 2.6438 s
// version 3, threads 12, n = 20000000: pi ~ 3.141592603589817, |error| = 5.000e-08, time = 2.0791 s
// version 4, threads 12, n = 20000000: pi ~ 3.141592603589847, |error| = 5.000e-08, time = 0.0089 s
// version 5, threads 12, n = 20000000: pi ~ 37.699111243077802, |error| = 3.456e+01, time = 0.0591 s
// :~/ 