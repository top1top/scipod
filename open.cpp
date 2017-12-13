#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <iomanip>
int N;
double *X;
double *ANS;
void inverse_gaissian() {

  for (int k = 0; k < N; k++)
    {

        double div = X[k*N + k];

            for (int j = 0; j < N; j++)
            {
                X[k*N + j] /= div;
                ANS[k*N + j] /= div;
            }
     
            for (int i = k + 1; i < N; i++)
            {
                double multi = X[i*N + k];


                for (int j = 0; j < N; j++)
                {
                    X[i*N + j] -= multi * X[k*N + j];
                    ANS[i*N + j] -= multi * ANS[k*N + j];
                }
            }
    }

    for (int k = N - 1; k > 0; k--)
    {
            for (int i = k - 1; i > -1; i--)
            {
                double multi = X[i*N + k];

                for (int j = 0; j < N; j++)
                {
                    X[i*N + j] -= multi * X[k*N +j];
                    ANS[i*N + j] -= multi * ANS[k*N + j];
                }
            }
    }
    
}
void inverse_gaissian_OPENMP()
{
//Трансформация исходной матрицы в верхнетреугольную
    for (int k = 0; k < N; k++)
    {

        double div = X[k*N + k];

#pragma omp parallel
        {
#pragma omp for
            for (int j = 0; j < N; j++)
            {
                X[k*N + j] /= div;
                ANS[k*N + j] /= div;
            }
        }
#pragma omp parallel
        {
#pragma omp for
            for (int i = k + 1; i < N; i++)
            {
                double multi = X[i*N + k];


                for (int j = 0; j < N; j++)
                {
                    X[i*N + j] -= multi * X[k*N + j];
                    ANS[i*N + j] -= multi * ANS[k*N + j];
                }
            }
        }
    }

    //Формирование единичной матрицы из исходной
    //и обратной из единичной
    for (int k = N - 1; k > 0; k--)
    {
#pragma omp parallel
        {
#pragma omp for
            for (int i = k - 1; i > -1; i--)
            {
                double multi = X[i*N + k];

                for (int j = 0; j < N; j++)
                {
                    X[i*N + j] -= multi * X[k*N +j];
                    ANS[i*N + j] -= multi * ANS[k*N + j];
                }
            }
        }
    }
    
}
void init()
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            X[i*N + j] = rand() % 1000;
            if (i == j) {
                ANS[i*N + j] = 1.0;
            } else {
                ANS[i*N + j] = 0.0;
            }
        }
    }
}
int main(){
    std::cin >> N;
    //N = 3;
    X = new double [N*N];
    ANS  = new double [N*N];
    init();
    /*X[0]= 1;
    X[1]= 3;
    X[2]= 3;
    X[3]= -2;
    X[4]= 3;
    X[5]= 2;
    X[6]= 2;
    X[7]= 3;
    X[8]= 2;

    ANS[0]= 1;
    ANS[1]= 0;
    ANS[2]= 0;
    ANS[3]= 0;
    ANS[4]= 1;
    ANS[5]= 0;
    ANS[6]= 0;
    ANS[7]= 0;
    ANS[8]= 1;*/
    double timerOpenMp = omp_get_wtime();
    inverse_gaissian();
    timerOpenMp = omp_get_wtime() - timerOpenMp;

    std::cout <<  " TIME :  " << std::setprecision(6) << std::fixed <<timerOpenMp << std::endl;

    /*for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << X[i*N + j] << " ";
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << ANS[i*N + j] << " ";
        }
        std::cout << std::endl;
    }*/
    return 0;
}