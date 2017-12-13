#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>

#include <iostream>

#define ROOT 0

using namespace std;

const int MAX_ABS_VALUE = 10;
const int MAGIC_NUMBER = 576;
const double EPS = 1E-3;

bool ensureMatrixIsInverse(double**, double**, int);

int main(int argc, char** argv) {
    int n;
    sscanf(argv[1],"%d",&n);
    double** a;
    double** b;
    double** raw;

    srand(time(NULL));

    double startTime;

    int rank, procCount;

    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    a = new double*[n];
    for (int i = 0; i < n; i++) {
        a[i] = new double[n];
    }

    b = new double*[n];
    for (int i = 0; i < n; i++) {
        b[i] = new double[n];
        for (int j = 0; j < n; j++) {
            b[i][j] = (i == j ? 1 : 0);
        }
    }

    if (rank == ROOT) {
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++) {
                a[i][j] = rand()%1000; 
            }
        }

        raw = new double*[n];
        for (int i = 0; i < n; i++) {
            raw[i] = new double[n];
            for (int j = 0; j < n; j++) {
                raw[i][j] = a[i][j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == ROOT) {
        startTime = MPI_Wtime();
    }

    for (int i = 0; i < n; i++) {
        MPI_Bcast(a[i], n, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }

    for (int i = 0; i < n; i++) {
        MPI_Bcast(b[i], n, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }

    double* shared_a = new double[n];
    double* shared_b = new double[n];

    // Starting Gaussian elimination.
    for (int row = rank; row < n; row += procCount) {
        for (int rcv_row = 0; rcv_row < row; rcv_row++) {
            int rcv_from = rcv_row % procCount;

            MPI_Recv(shared_a, n, MPI_DOUBLE, rcv_from, rcv_row * n + row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(shared_b, n, MPI_DOUBLE, rcv_from, rcv_row * n + row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // subtracting every received row.
            double mult = a[row][rcv_row];

            for (int i = 0; i < n; i++) {
                shared_a[i] *= mult;
                shared_b[i] *= mult;
                a[row][i] -= shared_a[i];
                b[row][i] -= shared_b[i];
            }
        }

        for (int i = 0; i < n; i++) {
            shared_a[i] = a[row][i];
        }
        double div_on = shared_a[row];
        for (int col = 0; col < n; col++) {
            shared_a[col] /= div_on;
            a[row][col] /= div_on;
            b[row][col] /= div_on;
            shared_b[col] = b[row][col];
        }

        for (int send_to_row = row + 1; send_to_row < n; send_to_row++) {
            MPI_Send(a[row], n, MPI_DOUBLE, send_to_row % procCount, row * n + send_to_row, MPI_COMM_WORLD);
            MPI_Send(shared_b, n, MPI_DOUBLE, send_to_row % procCount, row * n + send_to_row, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int start_from_row = n - 1;
    while (start_from_row % procCount != rank) {
        start_from_row--;
    }

    for (int row = start_from_row; row >= 0; row -= procCount) {
        for (int rcv_row = n - 1; rcv_row > row; rcv_row--) {
            int rcv_from = rcv_row % procCount;

            MPI_Recv(shared_a, n, MPI_DOUBLE, rcv_from, rcv_row * n + row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(shared_b, n, MPI_DOUBLE, rcv_from, rcv_row * n + row, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // subtracting every received row.
            double mult = a[row][rcv_row];

            for (int i = 0; i < n; i++) {
                shared_a[i] *= mult;
                shared_b[i] *= mult;
                a[row][i] -= shared_a[i];
                b[row][i] -= shared_b[i];
            }
        }

        for (int i = 0; i < n; i++) {
            shared_a[i] = a[row][i];
        }
        double div_on = shared_a[row];
        for (int col = 0; col < n; col++) {
            shared_a[col] /= div_on;
            a[row][col] /= div_on;
            b[row][col] /= div_on;
            shared_b[col] = b[row][col];
        }

        for (int send_to_row = row - 1; send_to_row >= 0; send_to_row--) {
            MPI_Send(a[row], n, MPI_DOUBLE, send_to_row % procCount, row * n + send_to_row, MPI_COMM_WORLD);
            MPI_Send(shared_b, n, MPI_DOUBLE, send_to_row % procCount, row * n + send_to_row, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == ROOT) {
        for (int row = 0; row < n; row++) {
            if (row % procCount != ROOT) {
                MPI_Recv(b[row], n, MPI_DOUBLE, row % procCount, MAGIC_NUMBER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        double duration = MPI_Wtime() - startTime;
        std::cout << "TIME:     " << duration << std::endl;
    } else {
        for (int row = 0; row < n; row++) {
            if (row % procCount != ROOT) {
                MPI_Send(b[row], n, MPI_DOUBLE, ROOT, MAGIC_NUMBER, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
