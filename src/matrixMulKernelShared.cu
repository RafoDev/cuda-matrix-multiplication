#include <iostream>
#include <string>
#include <vector>
#include "../include/utils.hpp"

using namespace std;

#define TILE_WIDTH 16

__global__ void MatrixMulKernelShared(float *M, float *N, float *P, int width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int ph = 0; ph < width / TILE_WIDTH; ph++)
    {
        Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + Col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }
    P[Row * width + Col] = Pvalue;
}


void launchKernel(int width, bool debug)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int size = width * width * sizeof(float);

    vector<float> M(size), N(size), P(size);

    generateRandomMatrix(M, width);
    generateRandomMatrix(N, width);

    float *h_M = M.data();
    float *h_N = N.data();
    float *h_P = P.data();

    float *d_M, *d_N, *d_P;

    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, h_P, size, cudaMemcpyHostToDevice);

    int blockDim = TILE_WIDTH;
    int gridDim = ceil(width / blockDim);

    dim3 dimGrid(gridDim, gridDim, 1);
    dim3 dimBlock(blockDim, blockDim, 1);

    cudaEventRecord(start);
    MatrixMulKernelShared<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
    cudaEventRecord(stop);

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    if (debug)
    {
        cout << "M: \n";
        printMatrix(M, width);
        cout << '\n';
        cout << "N: \n";
        printMatrix(N, width);
        cout << '\n';
        cout << "P: \n";
        printMatrix(P, width);
        cout << '\n';
    }
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    cudaEventSynchronize(stop);

    float miliseconds = 0;
    cudaEventElapsedTime(&miliseconds, start, stop);
    cout << "miliseconds (shared): " << miliseconds << '\n';
}

int main(int argc, char *argv[])
{
    bool debug = false;
    if (argc < 2)
    {
        cout << "Usage: matrixMultKernelShared <n> \n";
        return 0;
    }
    else if (argc == 3)
    {
        if (string(argv[2]) == "-DEBUG")
        {
            debug = true;
        }
        else
        {
            cout << "Usage: matrixMultKernelShared <n> -DEBUG\n";
            return 0;
        }
    }

    int width = stoi(argv[1]);

    launchKernel(width, debug);
}