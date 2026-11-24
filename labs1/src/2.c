/*
 * File:     histogram_starter.c
 * Purpose:  Starter code for Practice Task P1. The serial histogram is
 *           the reference; Histogram_par() is yours to write.
 * Compile:  gcc -O2 -Wall -fopenmp -o histogram histogram_starter.c
 * Run:      ./histogram <thread_count> <data_count> <bin_count>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Generate deterministic pseudo-random floats in [0, 5) */
void Gen_data(float* data, long data_count) {
   unsigned int s = 12345u;                     /* fixed seed: repeatable */
   for (long i = 0; i < data_count; i++) {
      s = 1103515245u * s + 12345u;             /* simple LCG */
      data[i] = 5.0f * (float)(s >> 8) / 16777216.0f;
   }
}

/* Map value x in [min, max) to a bin index in [0, bin_count - 1] */
int Which_bin(float x, float min, float max, int bin_count) {
   int b = (int)((x - min) / (max - min) * bin_count);
   return (b >= bin_count) ? bin_count - 1 : b;
}

/* Reference serial implementation */
void Histogram_serial(const float* data, long data_count, long* hist,
                      int bin_count) {
   memset(hist, 0, bin_count * sizeof(long));
   for (long i = 0; i < data_count; i++) {
      hist[Which_bin(data[i], 0.0f, 5.0f, bin_count)]++;
   }
}

/* TODO (P1): write three parallel versions -- see the task statement */
void Histogram_par(const float* data, long data_count, long* hist,
                   int bin_count, int thread_count) {
   memset(hist, 0, bin_count * sizeof(long));

#  pragma omp parallel for num_threads(thread_count)   /* BUGGY on purpose */
   for (long i = 0; i < data_count; i++) {
      hist[Which_bin(data[i], 0.0f, 5.0f, bin_count)]++;
   }
}

int main(int argc, char* argv[]) {
   if (argc != 4) {
      fprintf(stderr, "usage: %s <threads> <data_count> <bin_count>\n", argv[0]);
      return 1;
   }

   int  thread_count = (int)strtol(argv[1], NULL, 10);
   long data_count   = strtol(argv[2], NULL, 10);
   int  bin_count    = (int)strtol(argv[3], NULL, 10);

   float* data = malloc(data_count * sizeof(float));
   long*  h_s  = malloc(bin_count * sizeof(long));
   long*  h_p  = malloc(bin_count * sizeof(long));

   if (!data || !h_s || !h_p) {
      fprintf(stderr, "Memory allocation failed\n");
      free(data);
      free(h_s);
      free(h_p);
      return 1;
   }

   Gen_data(data, data_count);

   double t0 = omp_get_wtime();
   Histogram_serial(data, data_count, h_s, bin_count);
   double t_s = omp_get_wtime() - t0;

   t0 = omp_get_wtime();
   Histogram_par(data, data_count, h_p, bin_count, thread_count);
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



// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ls
// bin  Lab1_OpenMP.docx  Lab1_OpenMP.pdf  LABLOG.md  logs  results  src
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ gcc -O2 -Wall -fopenmp -o ./bin/c2 ./src/2.c -lm
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/c2 4 10000 10
// bins wrong: 10 of 10, counted 2700 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/c2 1 10000 10
// bins wrong: 0 of 10, counted 10000 of 10000 values
// verification: PASSED   T_serial = 0.0000 s   T_parallel = 0.0000 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/c2 3 10000 10
// bins wrong: 10 of 10, counted 3766 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/c2 6 10000 10
// bins wrong: 10 of 10, counted 2036 of 10000 values
// verification: FAILED   T_serial = 0.0000 s   T_parallel = 0.0002 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 










