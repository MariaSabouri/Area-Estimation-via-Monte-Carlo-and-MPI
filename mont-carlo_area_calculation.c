#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
/*
If your code includes mathematical functions (like exp, cos, etc.), you need to link to the mathematics 
library libm.so. This is done, just like for serial compiling, by adding -lm to the end of your compile command, 
that is,

mpicc mont-carlo_area_calculation.c -o mont-carlo_area_calculation -lm
*/
#define U(x) (sin(x) + 1)   //U(x) must be positive in my code 

double max_func(
    double a,
    double b
){
    double pi = acos(-1.0);

    double k_min = ceil((a - pi/2.0) / (2.0*pi));
    double k_max = floor((b - pi/2.0) / (2.0*pi));

    if (k_min <= k_max) {
        return 2;
    }else if (U(a) > U(b)) {
        return U(a);
    }else {
        return U(b);
    }
}

void mont_carlo(
    int rank_num,
    int comm_sz,
    double left_endpt,
    double right_endpt,
    double h,
    int local_n,
    int mont_carlo_n,
    double* total_area,
    double* local_area
){
    *local_area = 0; 
    for (int i = 0; i < local_n; i++) {
        int hit = 0;
        
        for (int j = 0; j < mont_carlo_n; j++) {
            double r_x_normalized = (double) rand() / RAND_MAX;
            double r_y_normalized = (double) rand() / RAND_MAX;
            
            double r_x = left_endpt + i * h + h * r_x_normalized;
            double r_y = max_func(left_endpt + i * h, left_endpt + (i + 1) * h ) * r_y_normalized;

            hit += (U(r_x) >= r_y) ? 1 : 0;
        }
        *local_area += max_func(left_endpt + i * h, left_endpt + (i + 1) * h ) * h * hit / mont_carlo_n;
    }
    MPI_Reduce(local_area, total_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    

}

void get_input(
    int rank_num,
    int comm_sz,
    double* a_p,
    double* b_p,
    int* n_p,
    int* mont_carlo_n,
    double* h
){
    if (rank_num == 0){
        printf("Enter a, b, n and mont_carlo_n\n");
        scanf("%lf %lf %d %d", a_p, b_p, n_p, mont_carlo_n);
    }
    *h = (*b_p - *a_p) / *n_p;

    MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mont_carlo_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

}

void distribute_local_values(
    int rank_num,
    int comm_size,
    double a,
    double b,
    double h,
    double* local_a,
    double* local_b,
    int n,
    int* local_n
){
    /*calculate local_n*/
    int sub_area;
    *local_n = n / comm_size;
    int remainder = n % comm_size;
    *local_n += (rank_num < remainder) ? 1 : 0;

    /*calculate local_a and local_b*/
    *local_a = a + rank_num * (*local_n) * h;
    *local_b = *local_a + (*local_n) * h;
}
    
int main(void){
    srand(time(NULL));

    int comm_size, rank_num, n, local_n, mont_carlo_n;
    double a, b, h, local_a, local_b, total_area, local_area;


    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_num);

    get_input(rank_num, comm_size, &a, &b, &n, &mont_carlo_n, &h);

    distribute_local_values(rank_num, comm_size, a, b, h, &local_a, &local_b, n, &local_n);

    mont_carlo(rank_num, comm_size, local_a, local_b, h, local_n, mont_carlo_n, &total_area, &local_area);

    total_area -= (b - a) * 1;
    if(rank_num == 0){
        printf("Total Area is %lf: \n", total_area);
    }

    MPI_Finalize();
    return 0;
}