// File: false_sharing.cpp
// Compile: g++ -O2 -Wall -fopenmp -o ./bin/false_sharing ./src/false_sharing.cpp
// Purpose:  Each thread sums its own block of an array into its own 
//            accumulator.  There is no data race - but where the 
//            accumulators live in memory decides the speed. 
//              version 0: double sums[t]          (neighbours share a cache line) 
//              version 1: sums padded to 64 bytes (one cache line each) 
//              version 2: private local variable, one write at the end
// Run: ./bin/false_sharing <thread_count> <n> <version 0-2>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using namespace std;

constexpr int CACHE_LINE = 64;

struct alignas(CACHE_LINE) PaddedDouble {
    double val = 0.0;
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "usage: " << argv[0] << " <thread_count> <n> <version 0-2>\n";
        return 1;
    }

    int thread_count = max(1, stoi(argv[1]));
    long n = stol(argv[2]);
    int version = stoi(argv[3]);

    vector<double> a(n, 1.0);
    vector<double> sums(thread_count, 0.0);
    vector<PaddedDouble> psums(thread_count);

    double total = 0.0;
    double start = omp_get_wtime();

#pragma omp parallel num_threads(thread_count) default(none) shared(a, n, version, sums, psums)
    {
        int t = omp_get_thread_num();

        if (version == 0) 
        {
            sums[t] = 0.0;
            #pragma omp for
            for (long i = 0; i < n; ++i)
                sums[t] += a[i];
        } 
        else if (version == 1) 
        {
            psums[t].val = 0.0;
            #pragma omp for
            for (long i = 0; i < n; ++i)
                psums[t].val += a[i];
        } 
        else 
        {
            double my_sum = 0.0;
            #pragma omp for
            for (long i = 0; i < n; ++i)
                my_sum += a[i];
            sums[t] = my_sum;
        }
    }

    double elapsed = omp_get_wtime() - start;

    for (int t = 0; t < thread_count; ++t)
        total += (version == 1) ? psums[t].val : sums[t];

    cout << "version " << version
         << ", threads " << setw(2) << thread_count
         << ": total = " << fixed << setprecision(1) << total
         << " (expected " << n << ")  "
         << "time = " << setprecision(4) << elapsed << " s\n";

    return 0;
}