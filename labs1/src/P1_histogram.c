/* File:     histogram_starter.c
 * Purpose:  Starter code for Practice Task P1. The serial histogram is
 *           the reference; Histogram_par() is yours to write.
 * Compile:  gcc -O2 -Wall -fopenmp -o histogram histogram_starter.c
 * Run:      ./histogram <thread_count> <data_count> <bin_count>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void Gen_data(float* data, long data_count) {   /* values in [0, 5) */
    unsigned int s = 12345u;                    /* fixed seed: repeatable */
    for (long i = 0; i < data_count; i++) {
        s = 1103515245u * s + 12345u;           /* simple LCG */
        data[i] = 5.0f * (float)(s >> 8) / 16777216.0f;
    }
}

int Which_bin(float x, float min, float max, int bin_count) {
    int b = (int)((x - min) / (max - min) * bin_count);
    return (b >= bin_count) ? bin_count - 1 : b;
}

void Histogram_serial(const float* data, long data_count, long* hist,
                      int bin_count) {
    memset(hist, 0, bin_count * sizeof(long));
    for (long i = 0; i < data_count; i++) {
        hist[Which_bin(data[i], 0.0f, 5.0f, bin_count)]++;
    }
}

/* (c) the OpenMP 4.5 array-section reduction reduction(+: hist[:bin_count]) */
void c_Histogram_par(const float* data, long data_count, long* hist, int bin_count, int thread_count) 
{

    memset(hist, 0, bin_count * sizeof(long));
    
    #pragma omp parallel for num_threads(thread_count) reduction(+: hist[:bin_count])

    for (long i = 0; i < data_count; i++) {
        hist[Which_bin(data[i], 0.0f, 5.0f, bin_count)]++;
    }
}

/* (a) #pragma omp atomic on the increment;*/
void a_Histogram_par(const float* data, long data_count, long* hist, 
    int bin_count, int thread_count) 
{
    memset(hist, 0, bin_count * sizeof(long));
    
    #pragma omp parallel for num_threads(thread_count)   /* BUGGY on purpose */
    for (long i = 0; i < data_count; i++) {
        #pragma omp atomic
        hist[Which_bin(data[i], 0.0f, 5.0f, bin_count)]++;
    }
}

/* (b) a private local_hist per thread, merged once at the end; */
void b_Histogram_par(const float* data, long data_count, long* hist, int bin_count, int thread_count) 
{
    memset(hist, 0, bin_count * sizeof(long));
    
    #pragma omp parallel num_threads(thread_count)
    {
        long* local_hist = malloc(bin_count * sizeof(long));

            #pragma omp for
            for (long i = 0; i < data_count; i++) {
                local_hist[Which_bin(data[i], 0.0f, 5.0f, bin_count)]++;
            }

            #pragma omp critical
            {
                for (int b = 0; b < bin_count; b++) {
                    hist[b] += local_hist[b];
                }
            }

            free(local_hist);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <threads> <data_count> <bin_count>\n", argv[0]);
        return 1;
    }

    int  thread_count = strtol(argv[1], NULL, 10);
    long data_count   = strtol(argv[2], NULL, 10);
    int  bin_count    = strtol(argv[3], NULL, 10);

    float* data = malloc(data_count * sizeof(float));
    long*  h_s  = malloc(bin_count * sizeof(long));
    long*  h_p  = malloc(bin_count * sizeof(long));

    Gen_data(data, data_count);

    double t0 = omp_get_wtime();
    Histogram_serial(data, data_count, h_s, bin_count);
    double t_s = omp_get_wtime() - t0;
    
    // printf("a_Histogram_par\n");
    printf("b_Histogram_par\n");
    // printf("c_Histogram_par\n");
    t0 = omp_get_wtime();
    b_Histogram_par(data, data_count, h_p, bin_count, thread_count);
    double t_p = omp_get_wtime() - t0;

    long bad = 0, total = 0;
    for (int b = 0; b < bin_count; b++) {
        if (h_s[b] != h_p[b]) bad++;
        total += h_p[b];
    }

    printf("bins wrong: %ld of %d, counted %ld of %ld values\n",
           bad, bin_count, total, data_count);
    printf("verification: %s   T_serial = %.4f s   T_parallel = %.4f s\n",
           bad == 0 ? "PASSED" : "FAILED", t_s, t_p);

    free(data);
    free(h_s);
    free(h_p);

    return bad != 0;
}





// // ======================= Before Correcting Algorithm ============================

// look to problem output only count value is issue here
 
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 1 10000 10
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0000 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 2 10000 10
// bins wrong: 10 of 10, counted 5505 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 3 10000 10
// bins wrong: 10 of 10, counted 3387 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0001 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 4 10000 10
// bins wrong: 10 of 10, counted 2689 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0003 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 6 10000 10
// bins wrong: 10 of 10, counted 2021 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 8 10000 10
// bins wrong: 10 of 10, counted 1864 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0003 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 10 10000 10
// bins wrong: 10 of 10, counted 1294 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0003 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 12 10000 10
// bins wrong: 10 of 10, counted 1059 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0003 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// // ============================================================================================






// ===============================================================================
// ======================= After Correcting Algorithm ============================
// ===============================================================================
//                          PART C
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 2 10000 10
// c_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0001 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 4 10000 10
// c_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0001 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 8 10000 10
// c_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 12 10000 10
// c_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0028 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// ===============================================================================
// ===============================================================================


// ===============================================================================
//                          PART A
// ===============================================================================
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 2 10000 10
// a_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0003 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 4 10000 10
// a_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0003 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 8 10000 10
// a_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0004 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 12 10000 10
// a_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0035 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// ===============================================================================
// ===============================================================================

// ===============================================================================
//                          PART B
// ===============================================================================
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 2 10000 10
// b_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 4 10000 10
// b_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 8 10000 10
// b_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0004 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/chistogram 12 10000 10
// b_Histogram_par
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0138 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// ===============================================================================
// ===============================================================================