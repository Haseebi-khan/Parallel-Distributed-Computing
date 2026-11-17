/* File:     histogram.cpp
 * Purpose:  SERIAL histogram (Pacheco & Malensek, Ch. 2 - Foster's methodology example).
 *           Baseline to be parallelized later with OpenMP.
 *
 * Input:    bin_count   number of bins
 *           min_meas    minimum value for the bin containing the smallest values
 *           max_meas    maximum value for the bin containing the largest values
 *           data_count  number of measurements
 *
 * Output:   bin_counts  number of measurements in each bin
 *
 * Bin rule: bin b holds   bin_maxes[b-1] <= measurement < bin_maxes[b]
 *           bin 0 holds   min_meas       <= measurement < bin_maxes[0]
 *
 * Compile:  g++ -O2 -Wall -Wextra -fopenmp -o ./bin/histogram ./src/histogram.cpp
 * Run:      ./bin/histogram <bin_count> <min_meas> <max_meas> <data_count>
 *           e.g. ./bin/histogram 5 0 5 20
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstdlib>
#include <omp.h>     // only for omp_get_wtime() in this serial version

using namespace std;

/* ------------------------------------------------------------------
 * Gen_data: fill data[] with uniform random floats in [min_meas, max_meas)
 * Fixed seed -> same data every run, so serial and parallel results
 * can be compared exactly.
 * ------------------------------------------------------------------ */
void Gen_data(vector<float>& data, float min_meas, float max_meas) {
    mt19937 gen(42);
    uniform_real_distribution<float> dist(min_meas, max_meas);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = dist(gen);
}

/* ------------------------------------------------------------------
 * Gen_bins: bin_width = (max_meas - min_meas) / bin_count
 *           bin_maxes[b] = min_meas + bin_width * (b + 1)
 * ------------------------------------------------------------------ */
void Gen_bins(vector<float>& bin_maxes, float min_meas, float max_meas) {
    int   bin_count = (int) bin_maxes.size();
    float bin_width = (max_meas - min_meas) / bin_count;
    for (int b = 0; b < bin_count; b++)
        bin_maxes[b] = min_meas + bin_width * (b + 1);
}

/* ------------------------------------------------------------------
 * Find_bin: linear search for the bin b that satisfies
 *           bin_maxes[b-1] <= value < bin_maxes[b]
 *           (bin_maxes[-1] is treated as min_meas)
 * ------------------------------------------------------------------ */
int Find_bin(float value, const vector<float>& bin_maxes, int bin_count, float min_meas) {
    for (int b = 0; b < bin_count; b++) {
        float lower = (b == 0) ? min_meas : bin_maxes[b - 1];
        if (lower <= value && value < bin_maxes[b])
            return b;
    }
    /* value == max_meas (or rounding put it just past the last max):
       put it in the last bin instead of losing it */
    return bin_count - 1;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "usage: " << argv[0]
             << " <bin_count> <min_meas> <max_meas> <data_count> <thread_num1>\n";
        return 1;
    }
    int   bin_count  = (int) strtol(argv[1], NULL, 10);
    float min_meas   = strtof(argv[2], NULL);
    float max_meas   = strtof(argv[3], NULL);
    long  data_count = strtol(argv[4], NULL, 10);
    int   T  = (int) strtol(argv[5], NULL, 10);
    

    if (bin_count <= 0 || data_count <= 0 || max_meas <= min_meas) {
        cerr << "need bin_count > 0, data_count > 0, max_meas > min_meas\n";
        return 1;
    }

    vector<float> data(data_count);
    vector<float> bin_maxes(bin_count);
    vector<int>   bin_counts(bin_count, 0);    // all counts start at 0

    Gen_data(data, min_meas, max_meas);
    Gen_bins(bin_maxes, min_meas, max_meas);

    /* ---------------- the part you will parallelize ---------------- */
    double start = omp_get_wtime();
    int* counts = bin_counts.data();
    #pragma omp parallel for num_threads(T) reduction(+: counts[:bin_count])
    for (long i = 0; i < data_count; i++) {
        int bin = Find_bin(data[i], bin_maxes, bin_count, min_meas);
        counts[bin]++;
    }

    double elapsed = omp_get_wtime() - start;
    /* --------------------------------------------------------------- */

    /* Print the histogram */
    cout << fixed << setprecision(3);
    long total = 0;
    for (int b = 0; b < bin_count; b++) {
        float lower = (b == 0) ? min_meas : bin_maxes[b - 1];
        cout << "[" << setw(8) << lower << ", " << setw(8) << bin_maxes[b] << ") : "
             << setw(10) << bin_counts[b];
        if (data_count <= 200) {                 // draw bars only for small inputs
            cout << "  ";
            for (int k = 0; k < bin_counts[b]; k++) cout << '*';
        }
        cout << "\n";
        total += bin_counts[b];
    }

    /* Verify: every measurement must land in exactly one bin */
    cout << "total counted = " << total << " / " << data_count
         << (total == data_count ? "  (PASSED)" : "  (FAILED)") << "\n";
    cout << setprecision(6) << "Time = " << elapsed << " s\n";

    return total != data_count;
}





// // Serial Program
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 10 0 100 20000000 1
// [   0.000,   10.000) :    2000439
// [  10.000,   20.000) :    1999157
// [  20.000,   30.000) :    1999777
// [  30.000,   40.000) :    1998130
// [  40.000,   50.000) :    2000847
// [  50.000,   60.000) :    2000178
// [  60.000,   70.000) :    2000889
// [  70.000,   80.000) :    1999685
// [  80.000,   90.000) :    2001466
// [  90.000,  100.000) :    1999432
// total counted = 20000000 / 20000000  (PASSED)
// Time = 1.494118 s


// // Parallel Program
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 10 0 100 20000000 8
// [   0.000,   10.000) :    2000439
// [  10.000,   20.000) :    1999157
// [  20.000,   30.000) :    1999777
// [  30.000,   40.000) :    1998130
// [  40.000,   50.000) :    2000847
// [  50.000,   60.000) :    2000178
// [  60.000,   70.000) :    2000889
// [  70.000,   80.000) :    1999685
// [  80.000,   90.000) :    2001466
// [  90.000,  100.000) :    1999432
// total counted = 20000000 / 20000000  (PASSED)
// Time = 0.270148 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 10 0 100 20000000 10
// [   0.000,   10.000) :    2000439
// [  10.000,   20.000) :    1999157
// [  20.000,   30.000) :    1999777
// [  30.000,   40.000) :    1998130
// [  40.000,   50.000) :    2000847
// [  50.000,   60.000) :    2000178
// [  60.000,   70.000) :    2000889
// [  70.000,   80.000) :    1999685
// [  80.000,   90.000) :    2001466
// [  90.000,  100.000) :    1999432
// total counted = 20000000 / 20000000  (PASSED)
// Time = 0.243035 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 10 0 100 20000000 12
// [   0.000,   10.000) :    2000439
// [  10.000,   20.000) :    1999157
// [  20.000,   30.000) :    1999777
// [  30.000,   40.000) :    1998130
// [  40.000,   50.000) :    2000847
// [  50.000,   60.000) :    2000178
// [  60.000,   70.000) :    2000889
// [  70.000,   80.000) :    1999685
// [  80.000,   90.000) :    2001466
// [  90.000,  100.000) :    1999432
// total counted = 20000000 / 20000000  (PASSED)
// Time = 0.214082 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ ./root 10 0 100 20000000 20
// [   0.000,   10.000) :    2000439
// [  10.000,   20.000) :    1999157
// [  20.000,   30.000) :    1999777
// [  30.000,   40.000) :    1998130
// [  40.000,   50.000) :    2000847
// [  50.000,   60.000) :    2000178
// [  60.000,   70.000) :    2000889
// [  70.000,   80.000) :    1999685
// [  80.000,   90.000) :    2001466
// [  90.000,  100.000) :    1999432
// total counted = 20000000 / 20000000  (PASSED)
// Time = 0.228907 s
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc$ 
