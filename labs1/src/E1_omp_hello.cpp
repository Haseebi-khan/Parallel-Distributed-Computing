#include <iostream>
#include <omp.h>

using namespace std;

void Hello(void)
{
    int my_rank      = omp_get_thread_num(); 
    int thread_count = omp_get_num_threads(); 
    printf("Hello from thread %d of %d\n", my_rank, thread_count); 
}

int main(int argc, char *argv[]){

    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <thread_count>\n", argv[0]);
        return 1;
    }

    int thread_count =  atoi(argv[1]);

    printf("Before the parallel region: %d thread(s) running\n", omp_get_num_threads());

    #pragma omp parallel num_threads(thread_count)
    Hello();   

    printf("After the parallel region: back to %d thread\n", omp_get_num_threads());


    return 0;
}

// without parallelism
// Before the parallel region: 6 thread(s) runing
// Hello from thread 0 of 1



// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ./bin/hello 1
// Before the parallel region: 12 thread(s) runing
// Hello from thread 0 of 1
// After the parallel region: back to 1 thread
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ./bin/hello 4
// Before the parallel region: 12 thread(s) runing
// Hello from thread 0 of 4
// Hello from thread 2 of 4
// Hello from thread 1 of 4
// Hello from thread 3 of 4
// After the parallel region: back to 1 thread
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ./bin/hello 8
// Before the parallel region: 12 thread(s) runing
// Hello from thread 4 of 8
// Hello from thread 2 of 8
// Hello from thread 0 of 8
// Hello from thread 3 of 8
// Hello from thread 5 of 8
// Hello from thread 1 of 8
// Hello from thread 6 of 8
// Hello from thread 7 of 8
// After the parallel region: back to 1 thread
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ./bin/hello 12
// Before the parallel region: 12 thread(s) runing
// Hello from thread 9 of 12
// Hello from thread 0 of 12
// Hello from thread 4 of 12
// Hello from thread 7 of 12
// Hello from thread 6 of 12
// Hello from thread 1 of 12
// Hello from thread 8 of 12
// Hello from thread 2 of 12
// Hello from thread 11 of 12
// Hello from thread 5 of 12
// Hello from thread 3 of 12
// Hello from thread 10 of 12
// After the parallel region: back to 1 thread
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ./bin/hello 16
// Before the parallel region: 12 thread(s) runing
// Hello from thread 2 of 16
// Hello from thread 8 of 16
// Hello from thread 13 of 16
// Hello from thread 12 of 16
// Hello from thread 6 of 16
// Hello from thread 11 of 16
// Hello from thread 1 of 16
// Hello from thread 0 of 16
// Hello from thread 14 of 16
// Hello from thread 3 of 16
// Hello from thread 7 of 16
// Hello from thread 9 of 16
// Hello from thread 4 of 16
// Hello from thread 10 of 16
// Hello from thread 5 of 16
// Hello from thread 15 of 16
// After the parallel region: back to 1 thread
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ OMP_DISPLAY_ENV=true ./bin/hello 2

// OPENMP DISPLAY ENVIRONMENT BEGIN
//   _OPENMP = '201511'
//   [host] OMP_DYNAMIC = 'FALSE'
//   [host] OMP_NESTED = 'FALSE'
//   [host] OMP_NUM_THREADS = '1'
//   [host] OMP_SCHEDULE = 'DYNAMIC'
//   [host] OMP_PROC_BIND = 'FALSE'
//   [host] OMP_PLACES = ''
//   [host] OMP_STACKSIZE = '0'
//   [host] OMP_WAIT_POLICY = 'PASSIVE'
//   [host] OMP_THREAD_LIMIT = '4294967295'
//   [host] OMP_MAX_ACTIVE_LEVELS = '1'
//   [host] OMP_NUM_TEAMS = '0'
//   [host] OMP_TEAMS_THREAD_LIMIT = '0'
//   [all] OMP_CANCELLATION = 'FALSE'
//   [all] OMP_DEFAULT_DEVICE = '0'
//   [all] OMP_MAX_TASK_PRIORITY = '0'
//   [all] OMP_DISPLAY_AFFINITY = 'FALSE'
//   [host] OMP_AFFINITY_FORMAT = 'level %L thread %i affinity %A'
//   [host] OMP_ALLOCATOR = 'omp_default_mem_alloc'
//   [all] OMP_TARGET_OFFLOAD = 'DEFAULT'
// OPENMP DISPLAY ENVIRONMENT END
// Before the parallel region: 12 thread(s) runing
// Hello from thread 0 of 2
// Hello from thread 1 of 2
// After the parallel region: back to 1 thread
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ 



// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// --------------------- Question ANswer ---------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------

// Q1:  Is the order of the Hello lines the same in every run? Why can’t it be?
// ANS:NO. because the threads are forked means divids into branches not queue as in squential.

// Q2: omp_get_num_threads() returns 1 before the region and 4 inside it. What does that tell you about 
// where the team exists?
// ANS: team exists only inside the parallel region. Before it, only the master runs as a single thread on a core inside parallel region
// forks a team of 4, and at end the threads join back into the master alone.

// Q3: Does ./omp_hello 16 work on a 4-core machine? What does it mean to have more threads than cores, 
// and why will that matter for timing? 
// ANS: yes, after runing the code then its on OS to give slicing time for each thread on core and complete the task
// but it create the extra work for OS. there will be no extra speed. 
// the 4 threads will run at same time but for other threads the OS will wait for long time to finish.
//  use more cache and context switching etc.


// Q4: Remove -fopenmp from the build command and rebuild with -Wall. What warning do you get, and what 
// does the program print?
// ANS: it shows the error that G++ doesnt known the pragmas which is kind of reference error 
// and the second error is about the thread count varible is not used.
// it can be handle macro and it program doesnt produced the exe file 

// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------


// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ g++ -g -Wall -o ./bin/hello ./src/E1_omp_hello.cpp
// ./src/E1_omp_hello.cpp:25: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]
//    25 |     #pragma omp parallel num_threads(thread_count)
//       | 
// ./src/E1_omp_hello.cpp: In function ‘int main(int, char**)’:
// ./src/E1_omp_hello.cpp:21:9: warning: unused variable ‘thread_count’ [-Wunused-variable]
//    21 |     int thread_count =  atoi(argv[1]);
//       |         ^~~~~~~~~~~~
// /usr/bin/ld: /tmp/ccZNfsxK.o: in function `Hello()':
// /home/noneo/A Semester/PDC/Code/pdc/labs1/./src/E1_omp_hello.cpp:8:(.text+0xd): undefined reference to `omp_get_thread_num'
// /usr/bin/ld: /home/noneo/A Semester/PDC/Code/pdc/labs1/./src/E1_omp_hello.cpp:9:(.text+0x15): undefined reference to `omp_get_num_threads'
// /usr/bin/ld: /tmp/ccZNfsxK.o: in function `main':
// /home/noneo/A Semester/PDC/Code/pdc/labs1/./src/E1_omp_hello.cpp:23:(.text+0x97): undefined reference to `omp_get_num_threads'
// /usr/bin/ld: /home/noneo/A Semester/PDC/Code/pdc/labs1/./src/E1_omp_hello.cpp:28:(.text+0xb7): undefined reference to `omp_get_num_threads'
// collect2: error: ld returned 1 exit status
// noneo@noneo-Precision-5530:~/A Semester/PDC/Code/pdc/labs$ ^C
