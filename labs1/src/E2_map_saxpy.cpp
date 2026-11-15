/* File:     map_saxpy.c
 * Purpose:  The map pattern: z[i] = a*x[i] + y[i].
 *           Serial baseline -> parallel for -> verify -> measure.
 * Compile:  gcc -O2 -Wall -fopenmp -o map_saxpy map_saxpy.c
 * Run:      ./map_saxpy <thread_count> <n> <reps>
 *           e.g. ./map_saxpy 4 50000000 10
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void saxpy(int n, float a, float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
      fprintf(stderr, "usage: %s <thread_count> <n> <reps>\n", argv[0]);
      return 1;
    }
    int  thread_count = (int) strtol(argv[1], NULL, 10);
    long n            = strtol(argv[2], NULL, 10);
    int  reps         = (int) strtol(argv[3], NULL, 10);
    const double a = 2.5;

    double* x  = (double*) malloc(n * sizeof(double));
    double* y  = (double*) malloc(n * sizeof(double));
    double* zs = (double*) malloc(n * sizeof(double));   /* serial result   */
    double* zp = (double*) malloc(n * sizeof(double));   /* parallel result */
    // Shows ERROR IF THE DATA IS NOT ALLOCATED DYNAMICALLY
    if (!x || !y || !zs || !zp) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Initialise in parallel with the same schedule as the compute loop:
       on multi-socket machines each page is placed near the thread that
       will use it ("first touch").                                       */
    #pragma omp parallel for num_threads(thread_count) default(none) shared(x, y, zs, zp, n)
    for (long i = 0; i < n; i++) 
    {
       x[i] = (double) (i % 1000);
       y[i] = 1.0 / (1.0 + (double) (i % 7));
       zs[i] = zp[i] = 0.0;
    }

    /* 1. Serial baseline: best of reps runs */
    double t_serial = 1e30;
    for (int r = 0; r < reps; r++) 
    {
        double start = omp_get_wtime();
        for (long i = 0; i < n; i++)
        {
            zs[i] = a*x[i] + y[i];
        }    
        double elapsed = omp_get_wtime() - start;
        if (elapsed < t_serial)
        {
            t_serial = elapsed;
        } 
    }

    /* 2. Parallel version: best of reps runs */
    double t_par = 1e30;
    for (int r = 0; r < reps; r++) 
    {
        double start = omp_get_wtime();
        #pragma omp parallel for num_threads(thread_count) default(none) shared(x, y, zp, n, a)
        for (long i = 0; i < n; i++)
        {
            zp[i] = a*x[i] + y[i];
        }   
        double elapsed = omp_get_wtime() - start;   /* after the barrier */
        if (elapsed < t_par){
            t_par = elapsed;
        } 
    }

    /* 3. Verify: same operations, same order per element -> exact match */
    long mismatches = 0;
    for (long i = 0; i < n; i++)
    {
        if (zp[i] != zs[i])
        {
            mismatches++;
        } 
    }

    /* 4. Measure: 2 loads + 1 store of 8 bytes per element */
    double bytes = 24.0 * (double) n;
    double S = t_serial / t_par;
    printf("n = %ld, threads = %d, reps = %d\n", n, thread_count, reps);
    printf("verification: %s (%ld mismatches)\n", mismatches == 0 ? "PASSED" : "FAILED", mismatches);
    printf("T_serial   = %.6f s  (%.2f GB/s)\n", t_serial, bytes/t_serial/1e9);
    printf("T_parallel = %.6f s  (%.2f GB/s)\n", t_par,    bytes/t_par/1e9);
    printf("speedup S  = %.2f   efficiency E = %.2f\n", S, S/thread_count);
    printf("CSV,map,%d,%ld,%.6f,%.6f,%.3f,%.3f\n", thread_count, n, t_serial, t_par, S, S/thread_count);

    free(x); free(y); free(zs); free(zp);
    return mismatches != 0;
}


// Output:

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ gcc -O2 -Wall -fopenmp -o ./bin/map_saxpy ./src/E2_map_saxpy.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/map_saxpy 1 20000000 5
// n = 20000000, threads = 1, reps = 5
// verification: PASSED (0 mismatches)
// T_serial   = 0.078629 s  (6.10 GB/s)
// T_parallel = 0.077906 s  (6.16 GB/s)
// speedup S  = 1.01   efficiency E = 1.01
// CSV,map,1,20000000,0.078629,0.077906,1.009,1.009

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/map_saxpy 2 20000000 5
// n = 20000000, threads = 2, reps = 5
// verification: PASSED (0 mismatches)
// T_serial   = 0.076469 s  (6.28 GB/s)
// T_parallel = 0.046998 s  (10.21 GB/s)
// speedup S  = 1.63   efficiency E = 0.81
// CSV,map,2,20000000,0.076469,0.046998,1.627,0.814

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/map_saxpy 4 20000000 5
// n = 20000000, threads = 4, reps = 5
// verification: PASSED (0 mismatches)
// T_serial   = 0.080529 s  (5.96 GB/s)
// T_parallel = 0.039059 s  (12.29 GB/s)
// speedup S  = 2.06   efficiency E = 0.52
// CSV,map,4,20000000,0.080529,0.039059,2.062,0.515


