/*
 * File:         jacobi_omp.cpp
 * Purpose:      2-D Jacobi iteration using OpenMP (P7)
 * P7:           One persistent parallel region for all iterations.
 *
 * Compile:      g++ -O2 -Wall -fopenmp -o jacobi_omp jacobi_omp.cpp
 * Run:          ./jacobi_omp <thread_count> <n> <max_iter> <tol>
 * Verification: ./jacobi_omp 4 37 5000 1e-6
 * Timing:       ./jacobi_omp 4 2048 500 0
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

using namespace std;


struct Grid {
    int n;
    int W;
    vector<double> data;

    explicit Grid(int n_val)
        : n(n_val), W(n_val + 2), data(static_cast<size_t>(W) * W, 0.0) {
        // Hot top edge = 100.0
        for (int j = 0; j < W; ++j) {
            (*this)(0, j) = 100.0;
        }
    }

    inline double& operator()(int i, int j) {
        return data[static_cast<size_t>(i) * W + j];
    }

    inline const double& operator()(int i, int j) const {
        return data[static_cast<size_t>(i) * W + j];
    }
};


int Jacobi_serial(Grid& u, Grid& u_new, int max_iter, double tol, double& diff_out) {
    int n = u.n;
    double diff = 0.0;
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        diff = 0.0;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                u_new(i, j) = 0.25 * (u(i - 1, j) + u(i + 1, j) +
                                      u(i, j - 1) + u(i, j + 1));
                diff = max(diff, abs(u_new(i, j) - u(i, j)));
            }
        }

        swap(u, u_new);

        if (diff < tol) {
            ++iter;
            break;
        }
    }

    diff_out = diff;
    return iter;
}


int Jacobi_omp(Grid& u, Grid& u_new, int max_iter, double tol, int thread_count, double& diff_out) {
    int n = u.n;
    double diff = 0.0;
    int iter = 0;

    int done = (max_iter <= 0);

    #pragma omp parallel num_threads(thread_count) default(none) \
        shared(u, u_new, n, max_iter, tol, diff, iter, done)
    {
        while (!done) 
        {
            #pragma omp single
            {
                diff = 0.0;
            }

            #pragma omp for reduction(max : diff)
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    u_new(i, j) = 0.25 * (u(i - 1, j) + u(i + 1, j) +
                                          u(i, j - 1) + u(i, j + 1));
                    diff = max(diff, abs(u_new(i, j) - u(i, j)));
                }
            }

            #pragma omp single
            {
                swap(u, u_new);
                ++iter;

                if (diff < tol || iter >= max_iter) {
                    done = 1;
                }
            }
        }
    }

    diff_out = diff;
    return iter;
}


int main(int argc, char* argv[]) 
{
    if (argc != 5) {
        cerr << "usage: " << argv[0] << " <thread_count> <n> <max_iter> <tol>\n";
        return 1;
    }

    int thread_count = stoi(argv[1]);
    int n            = stoi(argv[2]);
    int max_iter     = stoi(argv[3]);
    double tol       = stod(argv[4]);

    // Initialize boundary in BOTH grids.
    Grid us(n), us_new(n);
    Grid up(n), up_new(n);

    double ds = 0.0;
    double dp = 0.0;


    double t0 = omp_get_wtime();
    int it_s = Jacobi_serial(us, us_new, max_iter, tol, ds);
    double t_serial = omp_get_wtime() - t0;


    t0 = omp_get_wtime();
    int it_p = Jacobi_omp(up, up_new, max_iter, tol, thread_count, dp);
    double t_par = omp_get_wtime() - t0;


    double max_err = 0.0;
    for (int i = 0; i < n + 2; ++i) {
        for (int j = 0; j < n + 2; ++j) {
            max_err = max(max_err, abs(us(i, j) - up(i, j)));
        }
    }

    bool ok = (max_err == 0.0 && it_s == it_p && ds == dp);


    double S = (t_par > 0.0) ? t_serial / t_par : 0.0;
    double bytes = 16.0 * n * static_cast<double>(n) * it_p;


    printf("n = %d, threads = %d, iterations: serial %d, parallel %d\n",
           n, thread_count, it_s, it_p);

    printf("final diff: serial %.6e, parallel %.6e; centre value %.6f\n",
           ds, dp, up((n + 1) / 2, (n + 1) / 2));

    printf("verification: %s (max |serial - parallel| = %.3e)\n",
           ok ? "PASSED" : "FAILED", max_err);

    printf("T_serial = %.4f s, T_parallel = %.4f s, S = %.2f, E = %.2f\n",
           t_serial, t_par, S, thread_count > 0 ? S / thread_count : 0.0);

    if (it_p > 0 && t_par > 0.0) {
        printf("time/iteration = %.3f ms, effective bandwidth = %.2f GB/s\n",
               1e3 * t_par / it_p, bytes / t_par / 1e9);
    }

    printf("CSV,jacobi,%d,%d,%d,%.6f,%.6f,%.3f,%.3f\n",
           thread_count, n, it_p, t_serial, t_par, S,
           thread_count > 0 ? S / thread_count : 0.0);

    return ok ? 0 : 2;
}

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ g++ -O2 -Wall -fopenmp -o ./bin/jacobi ./src/P7_jocabi.cpp
// /usr/bin/ld: /usr/lib/gcc/x86_64-linux-gnu/13/../../../x86_64-linux-gnu/Scrt1.o: in function `_start':
// (.text+0x1b): undefined reference to `main'
// collect2: error: ld returned 1 exit status
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ g++ -O2 -Wall -fopenmp -o ./bin/jacobi ./src/P7_jocabi.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 4 37 5000 1e-6
// n = 37, threads = 4, iterations: serial 3462, parallel 3462
// final diff: serial 9.939494e-07, parallel 9.939494e-07; centre value 24.999709
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.0055 s, T_parallel = 0.0092 s, S = 0.60, E = 0.15
// time/iteration = 0.003 ms, effective bandwidth = 8.24 GB/s
// CSV,jacobi,4,37,3462,0.005492,0.009202,0.597,0.149
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 4 128 5000 1e-6
// n = 128, threads = 4, iterations: serial 5000, parallel 5000
// final diff: serial 2.929374e-03, parallel 2.929374e-03; centre value 16.123249
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.1082 s, T_parallel = 0.0390 s, S = 2.78, E = 0.69
// time/iteration = 0.008 ms, effective bandwidth = 33.63 GB/s
// CSV,jacobi,4,128,5000,0.108243,0.038973,2.777,0.694
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 4 2048 5000 1e-6
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 4 2048 5000 1e-6
// n = 2048, threads = 4, iterations: serial 5000, parallel 5000
// final diff: serial 4.839495e-03, parallel 4.839495e-03; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 46.9226 s, T_parallel = 36.6016 s, S = 1.28, E = 0.32
// time/iteration = 7.320 ms, effective bandwidth = 9.17 GB/s
// CSV,jacobi,4,2048,5000,46.922557,36.601634,1.282,0.320
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 


