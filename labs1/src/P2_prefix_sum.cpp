/*
 * File:     prefix_sum.cpp
 * Purpose:  Three-phase blocked parallel inclusive prefix sum (scan) in OpenMP.
 * Compile:  g++ -O2 -Wall -fopenmp -o prefix_sum prefix_sum.cpp
 * Run:      ./prefix_sum <thread_count> <n>
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <omp.h>

using namespace std;

/* Reference serial inclusive prefix sum */
template <typename T>
void scan_serial(const vector<T>& x, vector<T>& s) 
{
    const size_t n = x.size();

    if (n == 0) return;

    s[0] = x[0];
    for (size_t i = 1; i < n; ++i) {
        s[i] = s[i - 1] + x[i];
    }
}

/* Three-phase blocked parallel inclusive prefix sum */
template <typename T>
void scan_parallel(const vector<T>& x, vector<T>& s, int p) {
    const size_t n = x.size();

    if (n == 0) return;

    vector<T> block_totals(p, T(0));
    vector<T> block_offsets(p, T(0));

    #pragma omp parallel num_threads(p)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        /* Contiguous block partitioning handling n % p != 0 and n < p */
        const size_t base = n / nthreads;
        const size_t rem  = n % nthreads;
        const size_t start = tid * base + min<size_t>(tid, rem);
        const size_t count = base + (static_cast<size_t>(tid) < rem ? 1 : 0);
        const size_t end   = start + count;

        /* Phase 1: Local inclusive prefix sum within each thread's contiguous block */
        if (start < end) 
        {
            s[start] = x[start];
            for (size_t i = start + 1; i < end; ++i) {
                s[i] = s[i - 1] + x[i];
            }
            block_totals[tid] = s[end - 1];
        }
        #pragma omp barrier

        /* Phase 2: Exclusive prefix sum of block totals by a single thread */
        #pragma omp single
        {
            T running = T(0);
            for (int t = 0; t < nthreads; ++t) {
                block_offsets[t] = running;
                running += block_totals[t];
            }
        }
        /* Implicit barrier at end of #pragma omp single */

        /* Phase 3: Add block offset to every element in the thread's block */
        if (start < end && tid > 0) {
            const T offset = block_offsets[tid];
            for (size_t i = start; i < end; ++i) {
                s[i] += offset;
            }
        }
    }
}

/* Exact verification using long integers across standard and edge-case sizes */
bool verify_exact(int p) {
    const vector<size_t> test_sizes = {
        0, 1, 3, static_cast<size_t>(p - 1),
        static_cast<size_t>(p), static_cast<size_t>(p + 1),
        10007, 1000003
    };

    for (size_t n : test_sizes) {
        vector<long> x(n), s_serial(n, 0), s_par(n, 0);
        for (size_t i = 0; i < n; ++i) {
            x[i] = static_cast<long>((i % 7) - 3);
        }

        scan_serial(x, s_serial);
        scan_parallel(x, s_par, p);

        if (s_serial != s_par) {
            cerr << "Verification FAILED for n = " << n << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <thread_count> <n>\n";
        return 1;
    }

    const int p = atoi(argv[1]);
    const size_t n = strtoull(argv[2], nullptr, 10);

    if (p <= 0) {
        cerr << "Error: thread_count must be > 0\n";
        return 1;
    }

    /* 1. Exact integer verification */
    if (!verify_exact(p)) {
        return 1;
    }
    cout << "Exact long verification: PASSED (including n < p and n % p != 0)\n\n";

    /* 2. Performance timing with double data */
    vector<double> x(n, 1.0);
    vector<double> s_serial(n, 0.0);
    vector<double> s_par(n, 0.0);

    /* Warmup */
    scan_serial(x, s_serial);
    scan_parallel(x, s_par, p);

    const double t0 = omp_get_wtime();
    scan_serial(x, s_serial);
    const double t_serial = omp_get_wtime() - t0;

    const double t1 = omp_get_wtime();
    scan_parallel(x, s_par, p);
    const double t_par = omp_get_wtime() - t1;

    const double speedup = t_serial / t_par;
    const double efficiency = speedup / p;

    cout << fixed << setprecision(4);
    cout << "Benchmark (double, n = " << n << ", p = " << p << ")\n";
    cout << "  T_serial   = " << t_serial << " s\n";
    cout << "  T_parallel = " << t_par    << " s\n";
    cout << "  Speedup S  = " << speedup  << "\n";
    cout << "  Efficiency = " << efficiency * 100.0 << " %\n";

    return 0;
}








// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/prefix_sum 1 100000000
// Exact long verification: PASSED (including n < p and n % p != 0)

// Benchmark (double, n = 100000000, p = 1)
//   T_serial   = 0.1461 s
//   T_parallel = 0.1430 s
//   Speedup S  = 1.0217
//   Efficiency = 102.1656 %

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/prefix_sum 2 100000000
// Exact long verification: PASSED (including n < p and n % p != 0)

// Benchmark (double, n = 100000000, p = 2)
//   T_serial   = 0.1468 s
//   T_parallel = 0.1900 s
//   Speedup S  = 0.7727
//   Efficiency = 38.6339 %

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ g++ -O2 -Wall -fopenmp -o ./bin/prefix_sum ./src/P2_prefix_sum.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/prefix_sum 6 100000000
// Exact long verification: PASSED (including n < p and n % p != 0)

// Benchmark (double, n = 100000000, p = 6)
//   T_serial   = 0.1473 s
//   T_parallel = 0.2267 s
//   Speedup S  = 0.6499
//   Efficiency = 10.8309 %

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/prefix_sum 12 100000000
// Exact long verification: PASSED (including n < p and n % p != 0)

// Benchmark (double, n = 100000000, p = 12)
//   T_serial   = 0.1821 s
//   T_parallel = 0.2424 s
//   Speedup S  = 0.7512
//   Efficiency = 6.2601 %
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ^C