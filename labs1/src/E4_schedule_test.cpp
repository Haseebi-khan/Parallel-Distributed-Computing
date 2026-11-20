// File: schedule_test.cpp
// Compile: g++ -O2 -Wall -fopenmp -o schedule_test schedule_test.cpp
// Run: OMP_SCHEDULE="dynamic,4" ./schedule_test 4 20000

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using namespace std;

double f(long i) {
    long start = i * (i + 1) / 2;
    long finish = start + i;
    double val = 0.0;
    for (long j = start; j <= finish; ++j)
        val += sin(static_cast<double>(j));
    return val;
}

int main(int argc, char* argv[]) 
{
    if (argc != 3) {
        cerr << "usage: " << argv[0] << " <thread_count> <n>  \n";
        return 1;
    }

    int thread_count = max(1, stoi(argv[1]));
    long n = stol(argv[2]);

    vector<double> busy(thread_count, 0.0);
    double sum = 0.0;

    omp_sched_t kind;
    int chunk = 0;
    omp_get_schedule(&kind, &chunk);

    const char* names[] = {"?", "static", "dynamic", "guided", "auto"};
    int k = static_cast<int>(kind) & 0x7;

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(thread_count) default(none) shared(n, busy) reduction(+ : sum)
    {
        double t0 = omp_get_wtime();

        #pragma omp for schedule(runtime) nowait
        for (long i = 0; i <= n; ++i)
            sum += f(i);

        busy[omp_get_thread_num()] = omp_get_wtime() - t0;
    }

    double elapsed = omp_get_wtime() - start;

    double max_busy = 0.0, avg_busy = 0.0;
    for (double t : busy) {
        avg_busy += t / thread_count;
        max_busy = max(max_busy, t);
    }

    cout << "schedule = " << ((k >= 1 && k <= 4) ? names[k] : "?")
         << ", chunk = " << chunk
         << ", threads = " << thread_count
         << ", n = " << n << "\n"
         << fixed << setprecision(4);

    for (int t = 0; t < thread_count; ++t)
        cout << "  thread " << setw(2) << t << " busy " << busy[t] << " s\n";

    cout << "sum = " << scientific << setprecision(10) << sum
         << "   elapsed = " << fixed << setprecision(4) << elapsed << " s"
         << "   imbalance (max/avg) = " << setprecision(2) << (max_busy / avg_busy) << "\n";

    return 0;
}



// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ for s in "static" "static,1" "dynamic,1" "dynamic,64" "guided"; do     OMP_SCHEDULE="$s" ./bin/schedule_test 4 20000; done
// schedule = static, chunk = 0, threads = 4, n = 20000
//   thread  0 busy 0.1477 s
//   thread  1 busy 0.4928 s
//   thread  2 busy 1.0579 s
//   thread  3 busy 4.3379 s
// sum = 1.5082016445e+00   elapsed = 4.3381 s   imbalance (max/avg) = 2.87
// schedule = static, chunk = 1, threads = 4, n = 20000
//   thread  0 busy 1.5750 s
//   thread  1 busy 1.5250 s
//   thread  2 busy 1.5276 s
//   thread  3 busy 1.5678 s
// sum = 1.5082016445e+00   elapsed = 1.5753 s   imbalance (max/avg) = 1.02
// schedule = dynamic, chunk = 1, threads = 4, n = 20000
//   thread  0 busy 1.5359 s
//   thread  1 busy 1.5362 s
//   thread  2 busy 1.5359 s
//   thread  3 busy 1.5361 s
// sum = 1.5082016445e+00   elapsed = 1.5363 s   imbalance (max/avg) = 1.00
// schedule = dynamic, chunk = 64, threads = 4, n = 20000
//   thread  0 busy 1.5416 s
//   thread  1 busy 1.5870 s
//   thread  2 busy 1.5565 s
//   thread  3 busy 1.5686 s
// sum = 1.5082016445e+00   elapsed = 1.5872 s   imbalance (max/avg) = 1.02
// schedule = guided, chunk = 1, threads = 4, n = 20000
//   thread  0 busy 1.5849 s
//   thread  1 busy 1.5842 s
//   thread  2 busy 1.5844 s
//   thread  3 busy 1.5840 s
// sum = 1.5082016445e+00   elapsed = 1.5851 s   imbalance (max/avg) = 1.00
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 
