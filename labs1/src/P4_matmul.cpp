
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <omp.h>

using namespace std;

// 1a. Serial i-j-k
void matmul_serial_ijk(double** A, double** B, double** C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = 0.0;

            for (int k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 1b. Serial i-k-j
void matmul_serial_ikj(double** A, double** B, double** C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            double temp = A[i][k];

            for (int j = 0; j < n; j++)
            {
                C[i][j] += temp * B[k][j];
            }
        }
    }
}

// 2a. Parallel i-k-j
void matmul_par_ikj(double** A, double** B, double** C, int n, int p)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;
    }

    #pragma omp parallel for num_threads(p) schedule(static)
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            double temp = A[i][k];

            for (int j = 0; j < n; j++)
            {
                C[i][j] += temp * B[k][j];
            }
        }
    }
}

// 2b. Parallel i-j-k with collapse(2)
void matmul_par_ijk(double** A, double** B, double** C, int n, int p)
{
    #pragma omp parallel for collapse(2) num_threads(p) schedule(static)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }
}

// Verification
void verify(double** C_ref, double** C_test, int n, const char* label)
{
    double max_error = 0.0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double error = fabs(C_ref[i][j] - C_test[i][j]);

            if (error > max_error)
                max_error = error;
        }
    }

    cout << "Verify [" << label << "]: ";

    if (max_error < 1e-10)
        cout << "PASSED";
    else
        cout << "FAILED";

    cout << " (max error = " << scientific
         << max_error << ")" << fixed << endl;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cout << "Usage: ./matmul <threads> <n>" << endl;
        return 1;
    }

    int p = atoi(argv[1]);
    int n = atoi(argv[2]);

    if (p <= 0 || n <= 0)
    {
        cout << "Invalid input!" << endl;
        return 1;
    }

    omp_set_dynamic(0);

    // Allocate 2D arrays
    double** A = new double*[n];
    double** B = new double*[n];
    double** C_ref = new double*[n];
    double** C_test = new double*[n];

    for (int i = 0; i < n; i++)
    {
        A[i] = new double[n];
        B[i] = new double[n];
        C_ref[i] = new double[n];
        C_test[i] = new double[n];
    }

    // Initialize matrices
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = ((i + j) % 13) * 0.1;
            B[i][j] = ((i * 3 + j) % 17) * 0.1;
        }
    }

    // double flops = 2.0 * n * n * n;

    cout << "\nMatrix Multiplication (n = "
         << n << ", threads = " << p << ")\n\n";

    // Serial i-j-k
    double start = omp_get_wtime();

    matmul_serial_ijk(A, B, C_ref, n);

    double t_ser_ijk = omp_get_wtime() - start;

    // Serial i-k-j
    start = omp_get_wtime();

    matmul_serial_ikj(A, B, C_test, n);

    double t_ser_ikj = omp_get_wtime() - start;

    cout << fixed << setprecision(4);

    cout << "Serial i-j-k: " << t_ser_ijk << " seconds\n";

    cout << "Serial i-k-j: " << t_ser_ikj << " seconds\n";

    cout << "i-k-j speedup: "
         << t_ser_ijk / t_ser_ikj << "x\n";

    verify(C_ref, C_test, n, "Serial i-k-j");

    // Parallel i-k-j
    start = omp_get_wtime();

    matmul_par_ikj(A, B, C_test, n, p);

    double t_par_ikj = omp_get_wtime() - start;

    cout << "\nParallel i-k-j: "
         << t_par_ikj << " seconds\n";

    cout << "Speedup: " << t_ser_ikj / t_par_ikj << endl;

    cout << "Efficiency: "
         << (t_ser_ikj / t_par_ikj) / p * 100
         << "%\n";

    verify(C_ref, C_test, n, "Parallel i-k-j");

    // Parallel i-j-k
    start = omp_get_wtime();

    matmul_par_ijk(A, B, C_test, n, p);

    double t_par_ijk = omp_get_wtime() - start;

    cout << "\nParallel i-j-k: "
         << t_par_ijk << " seconds\n";

    cout << "Speedup: " << t_ser_ijk / t_par_ijk << endl;

    cout << "Efficiency: "
         << (t_ser_ijk / t_par_ijk) / p * 100
         << "%\n";

    verify(C_ref, C_test, n, "Parallel i-j-k");

    // Free memory
    for (int i = 0; i < n; i++)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C_ref[i];
        delete[] C_test[i];
    }

    delete[] A;
    delete[] B;
    delete[] C_ref;
    delete[] C_test;

    return 0;
}




// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/matmul 1 1000

// Matrix Multiplication (n = 1000, threads = 1)

// Serial i-j-k: 1.3910 seconds
// Serial i-k-j: 0.6524 seconds
// i-k-j speedup: 2.1323x
// Verify [Serial i-k-j]: PASSED (max error = 0.0000e+00)
// ^[[A^[[A
// Parallel i-k-j: 0.6209 seconds
// Speedup: 1.0507
// Efficiency: 105.0673%
// Verify [Parallel i-k-j]: PASSED (max error = 0.0000e+00)
// ^[[A^[[A
// Parallel i-j-k: 1.4150 seconds
// Speedup: 0.9830
// Efficiency: 98.3041%
// Verify [Parallel i-j-k]: PASSED (max error = 0.0000e+00)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/matmul 2 1000

// Matrix Multiplication (n = 1000, threads = 2)

// Serial i-j-k: 1.3802 seconds
// Serial i-k-j: 0.6598 seconds
// i-k-j speedup: 2.0920x
// Verify [Serial i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-k-j: 0.3280 seconds
// Speedup: 2.0113
// Efficiency: 100.5649%
// Verify [Parallel i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-j-k: 0.6775 seconds
// Speedup: 2.0372
// Efficiency: 101.8591%
// Verify [Parallel i-j-k]: PASSED (max error = 0.0000e+00)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/matmul 4 1000

// Matrix Multiplication (n = 1000, threads = 4)

// Serial i-j-k: 1.3934 seconds
// Serial i-k-j: 0.7501 seconds
// i-k-j speedup: 1.8576x
// Verify [Serial i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-k-j: 0.2124 seconds
// Speedup: 3.5323
// Efficiency: 88.3065%
// Verify [Parallel i-k-j]: PASSED (max error = 0.0000e+00)
// ^[[A^[[A
// Parallel i-j-k: 0.4036 seconds
// Speedup: 3.4522
// Efficiency: 86.3058%
// Verify [Parallel i-j-k]: PASSED (max error = 0.0000e+00)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/matmul 6 1000

// Matrix Multiplication (n = 1000, threads = 6)

// Serial i-j-k: 1.3643 seconds
// Serial i-k-j: 0.6158 seconds
// i-k-j speedup: 2.2154x
// Verify [Serial i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-k-j: 0.1378 seconds
// Speedup: 4.4705
// Efficiency: 74.5082%
// Verify [Parallel i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-j-k: 0.2320 seconds
// Speedup: 5.8812
// Efficiency: 98.0205%
// Verify [Parallel i-j-k]: PASSED (max error = 0.0000e+00)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/matmul 8 1000

// Matrix Multiplication (n = 1000, threads = 8)

// Serial i-j-k: 1.3508 seconds
// Serial i-k-j: 0.6085 seconds
// i-k-j speedup: 2.2200x
// Verify [Serial i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-k-j: 0.1219 seconds
// Speedup: 4.9910
// Efficiency: 62.3877%
// Verify [Parallel i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-j-k: 0.3753 seconds
// Speedup: 3.5994
// Efficiency: 44.9924%
// Verify [Parallel i-j-k]: PASSED (max error = 0.0000e+00)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/matmul 10 1000

// Matrix Multiplication (n = 1000, threads = 10)

// Serial i-j-k: 1.3796 seconds
// Serial i-k-j: 0.6156 seconds
// i-k-j speedup: 2.2409x
// Verify [Serial i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-k-j: 0.0999 seconds
// Speedup: 6.1651
// Efficiency: 61.6512%
// Verify [Parallel i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-j-k: 0.3867 seconds
// Speedup: 3.5672
// Efficiency: 35.6717%
// Verify [Parallel i-j-k]: PASSED (max error = 0.0000e+00)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ ./bin/matmul 12 1000

// Matrix Multiplication (n = 1000, threads = 12)

// Serial i-j-k: 1.3983 seconds
// Serial i-k-j: 0.7301 seconds
// i-k-j speedup: 1.9153x
// Verify [Serial i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-k-j: 0.1570 seconds
// Speedup: 4.6492
// Efficiency: 38.7434%
// Verify [Parallel i-k-j]: PASSED (max error = 0.0000e+00)

// Parallel i-j-k: 0.6371 seconds
// Speedup: 2.1948
// Efficiency: 18.2896%
// Verify [Parallel i-j-k]: PASSED (max error = 0.0000e+00)
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs1$ 