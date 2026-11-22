/* File:     jacobi_omp.cpp
 * Purpose:  2-D Jacobi iteration (5-point stencil) for the heat equation
 *           on an (n+2) x (n+2) grid -- the CCP common benchmark.
 * Compile:  g++ -O2 -Wall -fopenmp -o jacobi_omp jacobi_omp.cpp
 * Run:      ./jacobi_omp <thread_count> <n> <max_iter> <tol>
 *           Verification run:  ./jacobi_omp 4 37 5000 1e-6
 *           Timing run:        ./jacobi_omp 4 2048 500 0
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>
using namespace std;

/* Contiguous (n+2) x (n+2) 2D grid wrapper */
struct Grid {
    int n;
    int W;
    vector<double> data;

    explicit Grid(int n_val) : n(n_val), W(n_val + 2), data(static_cast<size_t>(W) * W, 0.0) 
        {
            // Set hot top edge (row 0 = 100.0); other boundaries & interior remain 0.0
            for (int j = 0; j < W; ++j) 
            {
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

/* ---------------- Serial Baseline ---------------- */
int Jacobi_serial(Grid& u, Grid& u_new, int max_iter, double tol, double& diff_out) {
    int n = u.n;
    double diff = 0.0;
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        diff = 0.0;
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                u_new(i, j) = 0.25 * (u(i - 1, j) + u(i + 1, j)
                                    + u(i, j - 1) + u(i, j + 1));
                diff = max(diff, abs(u_new(i, j) - u(i, j)));
            }
        }
        swap(u, u_new); // O(1) internal vector pointer swap
        if (diff < tol) {
            ++iter;
            break;
        }
    }

    diff_out = diff;
    return iter;
}

/* ---------------- OpenMP Version ---------------- */
int Jacobi_omp(Grid& u, Grid& u_new, int max_iter, double tol,
               int thread_count, double& diff_out) {
    int n = u.n;
    double diff = 0.0;
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        diff = 0.0;
        #pragma omp parallel for num_threads(thread_count) default(none) shared(u, u_new, n) reduction(max : diff)
        for (int i = 1; i <= n; ++i) 
        {
            for (int j = 1; j <= n; ++j) 
            {
                u_new(i, j) = 0.25 * (u(i - 1, j) + u(i + 1, j) + u(i, j - 1) + u(i, j + 1));
                diff = max(diff, abs(u_new(i, j) - u(i, j)));
            }
        }
        swap(u, u_new);
        if (diff < tol) 
        {
            ++iter;
            break;
        }
    }

    diff_out = diff;
    return iter;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "usage: " << argv[0] << " <thread_count> <n> <max_iter> <tol>\n";
        return 1;
    }

    int thread_count = stoi(argv[1]);
    int n            = stoi(argv[2]);
    int max_iter     = stoi(argv[3]);
    double tol       = stod(argv[4]);

    // Initialize boundary in BOTH grids (u and u_new)
    Grid us(n), us_new(n);
    Grid up(n), up_new(n);

    double ds = 0.0, dp = 0.0;
    // serial 
    double t0 = omp_get_wtime();
    int it_s = Jacobi_serial(us, us_new, max_iter, tol, ds);
    double t_serial = omp_get_wtime() - t0;

    // parallel 
    t0 = omp_get_wtime();
    int it_p = Jacobi_omp(up, up_new, max_iter, tol, thread_count, dp);
    double t_par = omp_get_wtime() - t0;

    // Bit-for-bit verification
    double max_err = 0.0;
    for (int i = 0; i < n + 2; ++i) {
        for (int j = 0; j < n + 2; ++j) {
            max_err = max(max_err, abs(us(i, j) - up(i, j)));
        }
    }

    bool ok = (max_err == 0.0 && it_s == it_p && ds == dp);
    double S = t_serial / t_par;
    double bytes = 16.0 * n * static_cast<double>(n) * it_p;

    printf("n = %d, threads = %d, iterations: serial %d, parallel %d\n",
                n, thread_count, it_s, it_p);
    printf("final diff: serial %.6e, parallel %.6e; centre value %.6f\n",
                ds, dp, up((n + 1) / 2, (n + 1) / 2));
    printf("verification: %s (max |serial - parallel| = %.3e)\n",
                ok ? "PASSED" : "FAILED", max_err);
    printf("T_serial = %.4f s, T_parallel = %.4f s, S = %.2f, E = %.2f\n",
                t_serial, t_par, S, S / thread_count);
    printf("time/iteration = %.3f ms, effective bandwidth = %.2f GB/s\n",
                1e3 * t_par / it_p, bytes / t_par / 1e9);
    printf("CSV,jacobi,%d,%d,%d,%.6f,%.6f,%.3f,%.3f\n",
                thread_count, n, it_p, t_serial, t_par, S, S / thread_count);

    return ok ? 0 : 2;
}


// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 2 5 100 1e-6
// n = 5, threads = 2, iterations: serial 100, parallel 100
// final diff: serial 3.146231e-06, parallel 3.146231e-06; centre value 24.999978
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.0000 s, T_parallel = 0.0002 s, S = 0.01, E = 0.01
// time/iteration = 0.002 ms, effective bandwidth = 0.17 GB/s
// CSV,jacobi,2,5,100,0.000003,0.000236,0.013,0.007



// ========================== Quesiton 5
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 4 37 5000 1e-6
// n = 37, threads = 4, iterations: serial 3462, parallel 3462
// final diff: serial 9.939494e-07, parallel 9.939494e-07; centre value 24.999709
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.0054 s, T_parallel = 0.0098 s, S = 0.55, E = 0.14
// time/iteration = 0.003 ms, effective bandwidth = 7.76 GB/s
// CSV,jacobi,4,37,3462,0.005360,0.009778,0.548,0.137
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// ========================== Quesiton 5



// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 3 100 20000 1e-8
// n = 100, threads = 3, iterations: serial 20000, parallel 20000
// final diff: serial 1.229850e-06, parallel 1.229850e-06; centre value 25.410580
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.2347 s, T_parallel = 0.1160 s, S = 2.02, E = 0.67
// time/iteration = 0.006 ms, effective bandwidth = 27.58 GB/s
// CSV,jacobi,3,100,20000,0.234711,0.116031,2.023,0.674
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// ========================== Quesiton 5





// ==================== Question 3 Analysis ==========================
// CSV,jacobi,1,1024,300,0.516573,0.477507,1.082,1.082
// CSV,jacobi,2,1448,300,0.985242,0.980859,1.004,0.502
// CSV,jacobi,4,2048,300,2.331486,1.925846,1.211,0.303
// CSV,jacobi,8,2896,300,4.067788,4.362125,0.933,0.117
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 1 1024 300 0
// n = 1024, threads = 1, iterations: serial 300, parallel 300
// final diff: serial 8.064433e-02, parallel 8.064433e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.5497 s, T_parallel = 0.4911 s, S = 1.12, E = 1.12
// time/iteration = 1.637 ms, effective bandwidth = 10.25 GB/s
// CSV,jacobi,1,1024,300,0.549715,0.491074,1.119,1.119

// ./bin/jacobi 2 1448 300 0
// n = 1448, threads = 2, iterations: serial 300, parallel 300
// final diff: serial 8.064433e-02, parallel 8.064433e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.9923 s, T_parallel = 0.8551 s, S = 1.16, E = 0.58
// time/iteration = 2.850 ms, effective bandwidth = 11.77 GB/s
// CSV,jacobi,2,1448,300,0.992338,0.855085,1.161,0.580

// ./bin/jacobi 4 2048 300 0
// n = 2048, threads = 4, iterations: serial 300, parallel 300
// final diff: serial 8.064433e-02, parallel 8.064433e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 2.0730 s, T_parallel = 1.8247 s, S = 1.14, E = 0.28
// time/iteration = 6.082 ms, effective bandwidth = 11.03 GB/s
// CSV,jacobi,4,2048,300,2.072983,1.824698,1.136,0.284

// ./bin/jacobi 8 2896 300 0
// n = 2896, threads = 8, iterations: serial 300, parallel 300
// final diff: serial 8.064433e-02, parallel 8.064433e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 3.9148 s, T_parallel = 4.3900 s, S = 0.89, E = 0.11
// time/iteration = 14.633 ms, effective bandwidth = 9.17 GB/s
// CSV,jacobi,8,2896,300,3.914777,4.390005,0.892,0.111
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// ==================================================================



// // ==================== Question 4 ==========================
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 1 2048 500 0
// n = 2048, threads = 1, iterations: serial 500, parallel 500
// final diff: serial 4.839573e-02, parallel 4.839573e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 3.4555 s, T_parallel = 3.2815 s, S = 1.05, E = 1.05
// time/iteration = 6.563 ms, effective bandwidth = 10.23 GB/s
// CSV,jacobi,1,2048,500,3.455459,3.281524,1.053,1.053


// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 2 2048 500 0
// n = 2048, threads = 2, iterations: serial 500, parallel 500
// final diff: serial 4.839573e-02, parallel 4.839573e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 3.5069 s, T_parallel = 3.1138 s, S = 1.13, E = 0.56
// time/iteration = 6.228 ms, effective bandwidth = 10.78 GB/s
// CSV,jacobi,2,2048,500,3.506857,3.113840,1.126,0.563
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 4 2048 500 0
// n = 2048, threads = 4, iterations: serial 500, parallel 500
// final diff: serial 4.839573e-02, parallel 4.839573e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 3.3781 s, T_parallel = 3.1920 s, S = 1.06, E = 0.26
// time/iteration = 6.384 ms, effective bandwidth = 10.51 GB/s
// CSV,jacobi,4,2048,500,3.378060,3.191967,1.058,0.265
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 8 2048 500 0
// n = 2048, threads = 8, iterations: serial 500, parallel 500
// final diff: serial 4.839573e-02, parallel 4.839573e-02; centre value 0.000000
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 3.3491 s, T_parallel = 3.2045 s, S = 1.05, E = 0.13
// time/iteration = 6.409 ms, effective bandwidth = 10.47 GB/s
// CSV,jacobi,8,2048,500,3.349070,3.204504,1.045,0.131
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
// // ==========================================================











// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 4 37 5000 1e-6
// n = 37, threads = 4, iterations: serial 3462, parallel 3462
// final diff: serial 9.939494e-07, parallel 9.939494e-07; centre value 24.999709
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.0054 s, T_parallel = 0.0069 s, S = 0.78, E = 0.20
// time/iteration = 0.002 ms, effective bandwidth = 10.95 GB/s
// CSV,jacobi,4,37,3462,0.005418,0.006926,0.782,0.196
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 

// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/jacobi 12 50 50000 1e-6
// n = 50, threads = 12, iterations: serial 5926, parallel 5926
// final diff: serial 9.998312e-07, parallel 9.998312e-07; centre value 25.817269
// verification: PASSED (max |serial - parallel| = 0.000e+00)
// T_serial = 0.0261 s, T_parallel = 1.5626 s, S = 0.02, E = 0.00
// time/iteration = 0.264 ms, effective bandwidth = 0.15 GB/s
// CSV,jacobi,12,50,5926,0.026088,1.562638,0.017,0.001
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 








































// PROBLEM UNDERSTANDING 


// To analyze this code clearly, it helps to separate it into two parts: **the physics/math problem being solved** and **how the code solves it both sequentially and in parallel**.

// ---

// ### 1. What is the Physical Problem? (The 2D Heat Equation)

// Imagine a flat, square metal plate.

// * You attach a heater to the **top edge**, keeping it permanently at **100°C**.
// * You pack ice around the **left, right, and bottom edges**, keeping them permanently at **0°C**.
// * Initially, the entire inside of the plate is at **0°C**.

// As time passes, heat flows from the hot top edge into the cold interior until the temperature across the plate stops changing (this final balanced state is called **steady state**).

// **The Goal:** Find the temperature at every point inside the plate once it reaches steady state.

// ---

// ### 2. How Do We Solve It on a Computer? (The 5-Point Stencil & Jacobi Method)

// A computer cannot calculate continuous temperatures everywhere, so we divide the metal plate into a grid of discrete cells—like a spreadsheet or chessboard.

// #### The Grid Setup: Why `(n + 2) x (n + 2)`?

// If we want an `n x n` grid of interior points whose temperatures can change, we wrap a 1-cell-thick border around it to hold the fixed boundary temperatures (`100.0` on top, `0.0` on the other 3 sides). That makes the total grid size `(n + 2) x (n + 2)`, indexed from `0` to `n + 1`:

// ```text
// Row 0:      [ 100.0 | 100.0 | 100.0 | 100.0 | 100.0 ]  <-- Fixed Hot Top Edge
// Row 1:      [  0.0  |  (1,1)|  (1,2)|  (1,3)|  0.0  ]  <-- Interior starts at i=1, j=1
// Row 2:      [  0.0  |  (2,1)|  (2,2)|  (2,3)|  0.0  ]
// Row 3:      [  0.0  |  (3,1)|  (3,2)|  (3,3)|  0.0  ]  <-- Interior ends at i=n, j=n
// Row n+1:    [  0.0  |  0.0  |  0.0  |  0.0  |  0.0  ]  <-- Fixed Cold Bottom Edge

// ```

// #### The Rule (5-Point Stencil)

// In physics, at steady state, the temperature of any interior point `(i, j)` is simply the **average of its 4 immediate neighbors** (North, South, West, East):

// $$u_{\text{new}}(i, j) = \frac{1}{4} \Big( u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) \Big)$$

// *(It's called a **5-point stencil** because it involves 1 center point and 4 neighbors).*

// #### Why Do We Need Two Grids (`u` and `u_new`)?

// When updating cell `(1, 2)`, you need the *old* temperature of `(1, 1)`, not the new temperature you just calculated a millisecond ago.

// * **`u`** holds the temperatures from the previous step (read-only during the calculation).
// * **`u_new`** stores the newly calculated averages.
// * Once the entire grid is updated, `swap(u, u_new)` makes the new grid become the current grid for the next round.

// #### When Do We Stop? (`tol` and `max_iter`)

// In every iteration, we track **`diff`**: the biggest temperature change that happened anywhere on the grid during that step (`abs(u_new(i, j) - u(i, j))`).

// * When `diff < tol` (for example, `1e-6`), temperatures are barely changing anymore—we have reached steady state and stop early.
// * If it hasn't converged by `max_iter` steps, we stop anyway (useful for timing benchmarks where `tol = 0` forces it to run all `max_iter` steps).

// ---

// ### 3. Mapping the Code Block-by-Block

// Now that you know the math, every piece of the C++ file has a clear purpose:

// #### A. `struct Grid` (Lines 21–40)

// * **What it does:** Represents the 2D `(n + 2) x (n + 2)` plate using a 1D `vector<double>` (`data`).
// * **Why 1D instead of `vector<vector<double>>`?** A 1D vector stores all numbers in one continuous block of RAM, which makes CPU cache access much faster.
// * **`operator()(int i, int j)`:** Translates 2D coordinates `(i, j)` into the 1D index `i * W + j`, letting the rest of the code write clean math like `u(i - 1, j)`.
// * **Constructor:** Fills the whole grid with `0.0`, then loops across row `0` setting `(*this)(0, j) = 100.0`.

// #### B. `Jacobi_serial` (Lines 43–64)

// * **What it does:** Solves the problem using **1 CPU core** as a baseline for speed and correctness.
// * **Key detail:** Notice the loops run from `1` to `n` (`for (int i = 1; i <= n; ++i)`). It intentionally skips row `0`, row `n+1`, column `0`, and column `n+1` so the boundary temperatures never get overwritten.

// #### C. `Jacobi_omp` (Lines 67–90)

// * **What it does:** Solves the exact same problem across **multiple CPU threads** using OpenMP.
// * **How it splits the work:**
// ```cpp
// #pragma omp parallel for num_threads(thread_count) default(none) \
//         shared(u, u_new, n) reduction(max : diff)

// ```


// * **`parallel for`:** Divides the rows (`i = 1` to `n`) evenly among `thread_count` threads. For example, with 4 threads and `n = 2048`, Thread 0 computes rows `1–512`, Thread 1 computes `513–1024`, etc.
// * **Why is this safe without locks?** Every thread reads from `u` (which nobody modifies) and writes to its own separate rows in `u_new`. Threads never collide.
// * **`reduction(max : diff)`:** Each thread tracks its own local maximum `diff` for its assigned rows. At the end of the `i` loop, OpenMP safely combines all local maximums into the single global `diff`.
// * **`swap(u, u_new)`:** Happens *outside* the parallel region on a single thread before the next iteration starts.



// #### D. `main` (Lines 92–147)

// `main` runs an experiment comparing the two functions and prints four key metrics:

// | Metric in Output | Formula / Meaning | What to Look For in Analysis |
// | --- | --- | --- |
// | **`verification: PASSED`** | Checks `max | us(i, j) - up(i, j) |
// | **`S` (Speedup)** | $S = \frac{T_{\text{serial}}}{T_{\text{parallel}}}$ | How many times faster the parallel code ran. With 4 threads, ideal speedup is close to `4.0`. |
// | **`E` (Parallel Efficiency)** | $E = \frac{S}{\text{thread\_count}}$ | Percentage of ideal hardware utilization (`1.00` = 100% efficient; drops if threads wait on memory or overhead). |
// | **`effective bandwidth`** | $\frac{16 \times n^2 \times \text{iterations}}{T_{\text{parallel}}}$ | Measures memory speed in GB/s (each cell reads 8 bytes from `u` and writes 8 bytes to `u_new` = 16 bytes per cell). Jacobi is **memory-bound**, meaning RAM speed usually limits speedup more than CPU math speed. |

// ---

// ### 4. How to Analyze the Two Example Runs in the Header

// Look at the two test commands written in the comment at the top of the file:

// 1. **Verification Run:** `./jacobi_omp 4 37 5000 1e-6`
// * **Small grid (`n = 37`):** Finishes very quickly.
// * **Strict tolerance (`tol = 1e-6`):** Tests whether the `diff < tol` stopping condition works and stops at the exact same iteration count in both serial and parallel.
// * *Analysis insight:* Speedup (`S`) will likely be poor or even `< 1.0` here because `37 x 37` is tiny—creating/synchronizing 4 threads 5,000 times costs more time than the math itself.


// 2. **Timing Run:** `./jacobi_omp 4 2048 500 0`
// * **Large grid (`n = 2048`, ~4.2 million cells):** Gives each thread heavy, meaningful work.
// * **Zero tolerance (`tol = 0`):** Since `diff` can never be `< 0`, it forces the code to run all `500` iterations so you get a fair, fixed-workload performance benchmark.
// * *Analysis insight:* Here you will see real speedup (`S`), though efficiency (`E`) will eventually plateau as you add more threads because all cores share the same RAM bandwidth.



// Would you like to walk through any specific part of the code in deeper detail, or discuss how to run experiments and write up an analysis of its scaling performance?



