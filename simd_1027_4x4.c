/*
201620890
Lee Sang Baek
2021/10/27 SIMD Programming
*/

#include <stdio.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <unistd.h>

#define ARRAY_SIZE 4

int arr_result[4][4] = {0,};

void print_matrix()
{
    for(int i=0;i<ARRAY_SIZE;i++)
    {
        for(int j=0;j<ARRAY_SIZE;j++)
        {
            printf("%6d", arr_result[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    int c;
    int opt;
    while((c = getopt(argc, argv, "v:")) != -1)
    {
        switch(c)
        {
            case 'v':
                opt = atoi(optarg);    
                break;
            case '?':
                printf("Option Error\n");
                printf("./matmul -v <option>\n");  
                exit(1);
                break;
        }
    }

    //4*4 matrix
    /*
    1234    4321
    5678    8765
    1357  x 7531
    2468    8642
    */
    int A[4][4] = {
                    {1,2,3,4},{5,6,7,8},
                    {1,3,5,7},{2,4,6,8}
                };
    int B[4][4] = {
                    {4,3,2,1},{8,7,6,5},
                    {7,5,3,1},{8,6,4,2}
                };
    
	
    int start, end;

    if(opt==0)
    {
        //Scalar
        int sum=0;

        start = __rdtsc();
        for(int i=0;i<ARRAY_SIZE;i++)
        {
            for(int j=0;j<ARRAY_SIZE;j++)
            {
                for(int k=0;k<ARRAY_SIZE;k++)
                {
                    arr_result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
		end = __rdtsc();
        printf("< Scalar Calculation >\n\n");
        print_matrix();
        printf("Elapsed time : %d\n", end-start);
    }
    else if(opt==1)
    {
        //SIMD Vector

        __m256i first, second, vec_res;
        __m256 mul_res, sum_res;
        int * result;

		start = __rdtsc();
        
        for(int i=0;i<ARRAY_SIZE;i++)
        {
            sum_res = _mm256_set1_ps(0.0);

            first = _mm256_setr_epi32(A[i][0],A[i][0],A[i][0],A[i][0],A[i][1],A[i][1],A[i][1],A[i][1]);
            second = _mm256_setr_epi32(B[0][0],B[0][1],B[0][2],B[0][3],B[1][0],B[1][1],B[1][2],B[1][3]);
            mul_res = _mm256_mul_ps(_mm256_cvtepi32_ps(first),_mm256_cvtepi32_ps(second));
            sum_res = _mm256_add_ps(sum_res, mul_res);
            
            first = _mm256_setr_epi32(A[i][2],A[i][2],A[i][2],A[i][2],A[i][3],A[i][3],A[i][3],A[i][3]);
            second = _mm256_setr_epi32(B[2][0],B[2][1],B[2][2],B[2][3],B[3][0],B[3][1],B[3][2],B[3][3]);
            mul_res = _mm256_mul_ps(_mm256_cvtepi32_ps(first),_mm256_cvtepi32_ps(second));
            sum_res = _mm256_add_ps(sum_res, mul_res);

            vec_res = _mm256_cvtps_epi32(sum_res);

            result = (int*)&vec_res;
            for(int m=0;m<ARRAY_SIZE;m++)
            {
                arr_result[i][m] = result[m]+result[m+4];
            }
        }
		end = __rdtsc();
        printf("< Vector Calculation >\n\n");
        print_matrix();
		printf("Elapsed time : %d\n", end-start);
    }
    else
    {
        printf("Option Error!\n");
        printf("-v 0 : Scalar Calculation\n");
        printf("-v 1 : Vector Calculation\n");
        exit(1);
    }
}
