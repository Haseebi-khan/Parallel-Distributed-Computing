#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

double Rsum(const double* a, long lo, long hi, long cutoff)
{
    if (hi - lo <= cutoff) {
        double s = 0.0;
        for (long i = lo; i < hi; i++)
            s += a[i];
        return s;
    }

    long mid = lo + (hi - lo) / 2;
    double left, right;

    #pragma omp task default(none) shared(left) firstprivate(a, lo, mid, cutoff)
    left = Rsum(a, lo, mid, cutoff);

    #pragma omp task default(none) shared(right) firstprivate(a, mid, hi, cutoff)
    right = Rsum(a, mid, hi, cutoff);

    #pragma omp taskwait

    return left + right;
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

    const long n = 100000000;
    vector<double> a(n, 1.0);

    omp_set_dynamic(0);
    omp_set_num_threads(p);

    /*
    long cutoffs[] = {10, 100, 1000, 10000, 100000, 1000000};

    for (long cutoff : cutoffs) {
        double total = 0.0;
        double start = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp single
            {
                total = Rsum(a.data(), 0, n, cutoff);
            }
        }

        double end = omp_get_wtime();

        cout << "Cutoff: " << cutoff
             << " | Time: " << end - start
             << " seconds | Sum: " << total << endl;
    }
    */

    double total = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for num_threads(p) reduction(+:total) schedule(static)
    for (long i = 0; i < n; i++) {
        total += a[i];
    }

    double end = omp_get_wtime();

    cout << "Reduction Time: " << end - start << " seconds\n";
    cout << "Sum: " << total << endl;

    return 0;
}



// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ g++ -O2 -Wall -fopenmp -o ./bin/rsum ./src/P6_RSum.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/rsum 4
// Cutoff: 10 | Time: 10.4686 seconds | Sum: 1e+08
// Cutoff: 100 | Time: 0.743148 seconds | Sum: 1e+08
// Cutoff: 1000 | Time: 0.123817 seconds | Sum: 1e+08
// Cutoff: 10000 | Time: 0.0578222 seconds | Sum: 1e+08
// Cutoff: 100000 | Time: 0.0521801 seconds | Sum: 1e+08
// Cutoff: 1000000 | Time: 0.0567166 seconds | Sum: 1e+08
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 




// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ g++ -O2 -Wall -fopenmp -o ./bin/rsum ./src/P6_RSum.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/rsum 4
// Reduction Time: 0.0504182 seconds
// Sum: 1e+08
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ g++ -O2 -Wall -fopenmp -o ./bin/rsum ./src/P6_RSum.cpp
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/rsum 4
// --- Recursive Tasks (p = 4) ---
// Cutoff:      10 | Time: 10.916685 s | Sum: 100000000.0
// Cutoff:     100 | Time: 0.653795 s | Sum: 100000000.0
// Cutoff:    1000 | Time: 0.125238 s | Sum: 100000000.0
// Cutoff:   10000 | Time: 0.063549 s | Sum: 100000000.0
// Cutoff:  100000 | Time: 0.062095 s | Sum: 100000000.0
// Cutoff: 1000000 | Time: 0.062879 s | Sum: 100000000.0

// --- Comparison (Best Cutoff vs. Reduction) ---
// Best Task Cutoff (100000): 0.062095 s (Sum: 100000000.0)
// Parallel For Reduction:      0.051884 s (Sum: 100000000.0)
// Reduction Speedup vs. Best Task: 1.20x