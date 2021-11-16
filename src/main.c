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

float arr_result[4][4] = {0,};

void print_matrix()
{
    for(int i=0;i<ARRAY_SIZE;i++)
    {
        for(int j=0;j<ARRAY_SIZE;j++)
        {
            printf("%6.1f", arr_result[i][j]);
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
                    {7,5,3,1},{8,6,4,2},
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


        __m256 one_vector_cvt;
        __m256 temp_sum_cvt;
        __m256 temp_row_cvt;
        float * temp_result;

        //make 2 dimension matrix
        int **mat_vec = aligned_alloc(32, sizeof(int)*ARRAY_SIZE);
        for(int i=0;i<ARRAY_SIZE;i++)
        {
            mat_vec[i] = aligned_alloc(32, sizeof(int)*ARRAY_SIZE);
        }

        //replace mat_vec with B matrix
        for(int i=0;i<ARRAY_SIZE;i++)
        {
            for(int j=0;j<ARRAY_SIZE;j++)
            {
                mat_vec[i][j] = B[i][j];
            }
        }        

		start = __rdtsc();
        for(int i=0;i<ARRAY_SIZE;i++)
        {
			temp_sum_cvt = _mm256_set1_ps(0.0);
            for(int j=0;j<ARRAY_SIZE;j++)
            {
                one_vector_cvt = _mm256_cvtepi32_ps(_mm256_set1_epi32(A[i][j]));
                temp_row_cvt = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)mat_vec[j]));
                temp_sum_cvt = _mm256_add_ps(temp_sum_cvt,_mm256_mul_ps(one_vector_cvt,temp_row_cvt));
            }
            temp_result = (float*)&temp_sum_cvt;
            for(int k=0;k<ARRAY_SIZE;k++)
            {
                arr_result[i][k] = temp_result[k];
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
