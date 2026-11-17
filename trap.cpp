/* File:     trap.cpp
 * Purpose:  Trapezoidal rule with OpenMP.
 *           Estimates the integral of f(x) from a to b using n trapezoids,
 *           then reports speedup S, efficiency E and serial fraction r.
 *
 * Method:   h   = (b - a) / n
 *           x_i = a + i*h,   i = 0, 1, ..., n
 *           Area ~= h * [ f(x_0)/2 + f(x_1) + ... + f(x_{n-1}) + f(x_n)/2 ]
 *
 * Metrics:  S = T_serial / T_parallel        (T_serial = same code with 1 thread)
 *           E = S / p
 *           r = (1/S - 1/p) / (1 - 1/p)      (serial fraction; Amdahl: S <= 1/r)
 *
 * Compile:  g++ -O2 -Wall -Wextra -fopenmp -o ./bin/trap ./src/trap.cpp
 * Run:      ./bin/trap <a> <b> <n> <num_threads>
 *           e.g. ./bin/trap 0 3 100000000 8
 */
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>

using namespace std;

/* ------------------------------------------------------------------
 * f: the function to integrate.  Change it to try other curves.
 * ------------------------------------------------------------------ */
double f(double x) {
    return x * x;
}

/* ------------------------------------------------------------------
 * Exact integral of f from a to b (for checking the answer).
 * For f(x) = x^2 the antiderivative is x^3 / 3.
 * Update this if you change f.
 * ------------------------------------------------------------------ */
double Exact(double a, double b) {
    return (b * b * b - a * a * a) / 3.0;
}

/* ------------------------------------------------------------------
 * Trap: trapezoidal rule with t threads
 *   - the two end points are counted with weight 1/2
 *   - the n-1 interior points x_1 ... x_{n-1} with weight 1
 * ------------------------------------------------------------------ */
double Trap(double a, double b, long n, int t) {

    double h = (b - a) / n;

    double approx = (f(a) + f(b)) / 2.0;     // f(x_0)/2 + f(x_n)/2

    #pragma omp parallel for num_threads(t) reduction(+:approx)
    for (long i = 1; i <= n - 1; i++)
    {      // interior points
        double x_i = a + i * h;
        approx += f(x_i);
    }

    return h * approx;
}

int main(int argc, char* argv[])
{
    if (argc != 5) {
        cerr << "usage: " << argv[0] << " <a> <b> <n> <num_threads>\n";
        return 1;
    }

    double a = strtod(argv[1], NULL);
    double b = strtod(argv[2], NULL);
    long   n = strtol(argv[3], NULL, 10);
    int    t = (int) strtol(argv[4], NULL, 10);
    if (n <= 0 || t <= 0) {
        cerr << "need n > 0 and num_threads > 0\n";
        return 1;
    }

    /* Warm-up: create the thread pool so neither timing pays for it */
    #pragma omp parallel num_threads(t)
    { }

    /* Serial baseline: the same code with 1 thread */
    double start      = omp_get_wtime();
    double area_s     = Trap(a, b, n, 1);
    double t_serial   = omp_get_wtime() - start;

    /* Parallel run: t threads */
    start             = omp_get_wtime();
    double area       = Trap(a, b, n, t);
    double t_parallel = omp_get_wtime() - start;

    /* Performance metrics */
    double S = t_serial / t_parallel;                  // speedup
    double E = S / t;                                  // efficiency

    double exact = Exact(a, b);
    cout << setprecision(15);
    cout << "With n = " << n << " trapezoids, the integral from "
         << a << " to " << b << "\n";
    cout << "serial   = " << area_s << "\n";
    cout << "estimate = " << area  << "\n";
    cout << "exact    = " << exact << "\n";
    cout << "error    = " << fabs(area - exact) << "\n";

    cout << fixed << setprecision(6);
    cout << "\nthreads (p)       = " << t << "\n";
    cout << "T_serial          = " << t_serial   << " s\n";
    cout << "T_parallel        = " << t_parallel << " s\n";
    cout << setprecision(3);
    cout << "speedup S         = " << S << "\n";
    cout << "efficiency E      = " << E << "\n";

    if (t > 1) {
        double r = (1.0 / S - 1.0 / t) / (1.0 - 1.0 / t);   // serial fraction
        cout << "serial fraction r = " << r << "\n";
        if (r > 0)
            cout << "Amdahl bound 1/r  = " << 1.0 / r << "  (max possible speedup)\n";
    } else {
        cout << "serial fraction r = n/a (needs p > 1)\n";
    }

    return 0;
}







// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 0 10 800000000 2
// With n = 800000000 trapezoids, the integral from 0 to 10
// serial   = 333.333333333305
// estimate = 333.333333333327
// exact    = 333.333333333333
// error    = 6.25277607468888e-12

// threads (p)       = 2
// T_serial          = 0.827837 s
// T_parallel        = 0.401176 s
// speedup S         = 2.064
// efficiency E      = 1.032
// serial fraction r = -0.031
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 0 10 800000000 4
// With n = 800000000 trapezoids, the integral from 0 to 10
// serial   = 333.333333333305
// estimate = 333.333333333326
// exact    = 333.333333333333
// error    = 6.87805368215777e-12

// threads (p)       = 4
// T_serial          = 0.811154 s
// T_parallel        = 0.216935 s
// speedup S         = 3.739
// efficiency E      = 0.935
// serial fraction r = 0.023
// Amdahl bound 1/r  = 43.004  (max possible speedup)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 0 10 800000000 6
// With n = 800000000 trapezoids, the integral from 0 to 10
// serial   = 333.333333333305
// estimate = 333.333333333342
// exact    = 333.333333333333
// error    = 8.35598257253878e-12

// threads (p)       = 6
// T_serial          = 0.814614 s
// T_parallel        = 0.174649 s
// speedup S         = 4.664
// efficiency E      = 0.777
// serial fraction r = 0.057
// Amdahl bound 1/r  = 17.460  (max possible speedup)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 0 10 800000000 8
// With n = 800000000 trapezoids, the integral from 0 to 10
// serial   = 333.333333333305
// estimate = 333.333333333295
// exact    = 333.333333333333
// error    = 3.84261511499062e-11

// threads (p)       = 8
// T_serial          = 0.802196 s
// T_parallel        = 0.149907 s
// speedup S         = 5.351
// efficiency E      = 0.669
// serial fraction r = 0.071
// Amdahl bound 1/r  = 14.142  (max possible speedup)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 0 10 800000000 10
// With n = 800000000 trapezoids, the integral from 0 to 10
// serial   = 333.333333333305
// estimate = 333.333333333325
// exact    = 333.333333333333
// error    = 8.12860889709555e-12

// threads (p)       = 10
// T_serial          = 0.835310 s
// T_parallel        = 0.121311 s
// speedup S         = 6.886
// efficiency E      = 0.689
// serial fraction r = 0.050
// Amdahl bound 1/r  = 19.899  (max possible speedup)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ 