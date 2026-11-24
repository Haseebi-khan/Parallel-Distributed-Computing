/*
 * File:     P5_pi.cpp
 * Purpose:  Parallel Monte Carlo estimation of Pi using thread-safe PRNGs.
 * Compile:  g++ -O2 -Wall -fopenmp -o ./bin/pi ./src/P5_pi.cpp
 * Run:      ./bin/pi <threads>
 */

#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

static inline double Next_uniform(unsigned long long* s) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return (*s >> 11) * (1.0 / 9007199254740992.0);
}

/* Monte Carlo Pi estimation using per-thread xorshift64 */
double estimate_pi(long long N, int p, long long* out_inside, double* out_time) {
    long long inside = 0;
    double start = omp_get_wtime();

    #pragma omp parallel num_threads(p) reduction(+:inside)
    {
        int my_rank = omp_get_thread_num();
        unsigned long long s = 88172645463325252ULL + 7919ULL * (my_rank + 1);

        #pragma omp for schedule(static)
        for (long long i = 0; i < N; i++) {
            double x = Next_uniform(&s);
            double y = Next_uniform(&s);
            if (x * x + y * y <= 1.0) {
                inside++;
            }
        }
    }

    *out_time = omp_get_wtime() - start;
    *out_inside = inside;
    return 4.0 * static_cast<double>(inside) / static_cast<double>(N);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <threads>\n";
        return 1;
    }

    int p = strtol(argv[1], nullptr, 10);
    if (p <= 0) {
        cerr << "Error: threads must be > 0\n";
        return 1;
    }

    const long long N = 10000000;
    long long inside = 0;
    double elapsed = 0.0;

    /* Run the xorshift64 version */
    double pi = estimate_pi(N, p, &inside, &elapsed);

    /*
    // Previous rand_r() version:
    double start = omp_get_wtime();

    #pragma omp parallel num_threads(p) reduction(+:inside)
    {
        unsigned int seed = 1u + 1337u * omp_get_thread_num();

        #pragma omp for
        for (long long i = 0; i < N; i++)
        {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;

            if (x * x + y * y <= 1.0)
            {
                inside++;
            }
        }
    }

    double end = omp_get_wtime();
    elapsed = end - start;
    double pi = 4.0 * inside / N;
    */

    cout << "Threads: " << p << "\n";
    cout << "Total points: " << N << "\n";
    cout << "Inside points: " << inside << "\n";
    cout << "Estimated PI: " << pi << "\n";
    cout << "Execution time: " << elapsed << " seconds\n";

    return 0;
}



// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ g++ -O2 -Wall -fopenmp -o ./bin/pi ./src/P5_pi.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 1
// Threads: 1
// Total points: 10000000
// Inside points: 7855124
// Estimated PI: 3.14205
// Execution time: 0.0830792 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 2
// Threads: 2
// Total points: 10000000
// Inside points: 7852497
// Estimated PI: 3.141
// Execution time: 0.0422854 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 4
// Threads: 4
// Total points: 10000000
// Inside points: 7853764
// Estimated PI: 3.14151
// Execution time: 0.0236074 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 12
// Threads: 12
// Total points: 10000000
// Inside points: 7853488
// Estimated PI: 3.1414
// Execution time: 0.0197109 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 











// ======================== 3 =================================================

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 1
// Threads: 1
// Total points: 10000000
// Inside points: 7853158
// Estimated PI: 3.14126
// Execution time: 0.0384828 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 2
// Threads: 2
// Total points: 10000000
// Inside points: 7853942
// Estimated PI: 3.14158
// Execution time: 0.0214023 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 4
// Threads: 4
// Total points: 10000000
// Inside points: 7853720
// Estimated PI: 3.14149
// Execution time: 0.0119506 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/pi 12
// Threads: 12
// Total points: 10000000
// Inside points: 7854078
// Estimated PI: 3.14163
// Execution time: 0.0144379 seconds
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ^C
// ======================== 3 =================================================