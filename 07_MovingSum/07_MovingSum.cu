/*
 * This program will calculate a moving sum over an vector, where each output is
 * the accumulation of a sliding window. In other words, the input for each
 * output is defined by the same global index but also accumulates the items left
 * and right of it of the size RADIUS.
 *
 *    ┌─────────────────────────────┐
 *    │1 1 1 1 1 1 1 1 1 1 1 1 1 1 1│
 *    └▲─▲─────▲─▲─▲───▲─▲─▲────────┘
 *     │ │     │ │ │   │ │ │
 *     ├─┘     └─┼─┘   └─┼─┘
 *     │         │       │-->RADIUS=1
 *     │         │       │
 *     │         │       │
 *    ┌┴─────────┴───────┴───────────┐
 *    │2 3 3 3 3 3 3 3 3 3 3 3 3 3 2 │
 *    └──────────────────────────────┘
 *    * Only three sliding window are drawn here, not all of them
 *
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

using namespace std;

#define RADIUS 10
#define BLOCKSIZE 512
#define WIDTH 65536

/* This is our CUDA call wrapper, we will use in PAC.
 *
 *  Almost all CUDA calls should be wrapped with this makro.
 *  Errors from these calls will be catched and printed on the console.
 *  If an error appears, the program will terminate.
 *
 * Example: gpuErrCheck(cudaMalloc(&deviceA, N * sizeof(int)));
 *          gpuErrCheck(cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
 */
#define gpuErrCheck(ans)                      \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}

/*
 *
 * Rewrite the kernel movingSumGlobal using a static allocation of shared
 * memory, as shown in the slides of the lecture.
 *
 */
__global__ void movingSumSharedMemStatic(int *vec, int *result_vec, int size)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shm[BLOCKSIZE + 2 * RADIUS];
    // One left for padding left and right
    int localIdx = threadIdx.x + RADIUS;
    shm[localIdx] = vec[globalIdx];

    // Left boundary reached
    if (threadIdx.x < RADIUS)
    {
        // Can take element from left
        if (globalIdx - RADIUS >= 0)
            shm[localIdx - RADIUS] = vec[globalIdx - RADIUS];
        // Fully on the left
        else
            shm[localIdx - RADIUS] = 0;
        // Can take element from right
        if (globalIdx + blockDim.x < size)
            shm[localIdx + blockDim.x] = vec[globalIdx + blockDim.x];
        // Fully on the right
        else
            shm[localIdx + blockDim.x] = 0;
    }

    __syncthreads();

    int result = 0;
    // Can even write to output
    if (globalIdx < size)
    {
        // From RADIUS elements left to RADIUS elements right of actual
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
        {
            // Accumulate result
            result += shm[localIdx + offset];
        }
        // Store result in global_memory
        result_vec[globalIdx] = result;
    }
}

/*
 *
 * Rewrite the kernel movingSumGlobal using a dynamic allocation of shared
 * memory, more information can be found here:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared
 *
 * You should use something like this:
 *    extern __shared__ int shmVec[];
 * and extend the execution configuration with the size of the shmVec.
 *
 */
__global__ void movingSumSharedMemDynamic(int *vec, int *result_vec, int size)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Dynamic shared memory allocation
    extern __shared__ int shm[];
    // One left for padding left and right
    int localIdx = threadIdx.x + RADIUS;
    shm[localIdx] = vec[globalIdx];

    // Left boundary reached
    if (threadIdx.x < RADIUS)
    {
        // Can take element from left
        if (globalIdx - RADIUS >= 0)
            shm[localIdx - RADIUS] = vec[globalIdx - RADIUS];
        // Fully on the left
        else
            shm[localIdx - RADIUS] = 0;
        // Can take element from right
        if (globalIdx + blockDim.x < size)
            shm[localIdx + blockDim.x] = vec[globalIdx + blockDim.x];
        // Fully on the right
        else
            shm[localIdx + blockDim.x] = 0;
    }

    __syncthreads();

    int result = 0;
    // Can even write to output
    if (globalIdx < size)
    {
        // From RADIUS elements left to RADIUS elements right of actual
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
        {
            // Accumulate result
            result += shm[localIdx + offset];
        }
        // Store result in global_memory
        result_vec[globalIdx] = result;
    }
}

/*
 *
 * Rewrite the kernel movingSumGlobal using only global memory and no
 * shared mem. Use atomic add operations.
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 *
 * You must reverse the access pattern, such that without atomics, conflicts can occure.
 * So 1 Thread writes "its" value into multiple outputs.
 */
__global__ void movingSumAtomics(int *vec, int *result_vec, int size)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
    {
        int idx = globalIdx + offset;
        // Check if the index does not exceed result_vec boundaries
        if (idx >= 0 && idx < size)
        {
            atomicAdd(&result_vec[idx], vec[globalIdx]);
        }
    }
}

// This is the GPU refernece implementation
__global__ void movingSumGlobal(int *vec, int *result_vec, int size)
{
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int result = 0;
    if (globalIdx >= RADIUS && globalIdx < size - RADIUS)
    {
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
        {
            result += vec[globalIdx + offset];
        }
    }

    if (globalIdx < RADIUS)
    {
        for (int offset = 0 - globalIdx; offset <= RADIUS; offset++)
        {
            result += vec[globalIdx + offset];
        }
    }

    if (globalIdx < size && globalIdx >= size - RADIUS)
    {
        for (int offset = -RADIUS; offset < size - globalIdx; offset++)
        {
            result += vec[globalIdx + offset];
        }
    }

    result_vec[globalIdx] = result;
}

// CPU reference implementation
void movingSumCPU(int *vec, int *result_vec, int size)
{
    int result;

    for (int i = 0; i < size; ++i)
    {
        result = 0;

        if (i >= RADIUS && i < size - RADIUS)
        {
            for (int offset = -RADIUS; offset <= RADIUS; offset++)
            {
                result += vec[i + offset];
            }
        }

        if (i < RADIUS)
        {
            for (int offset = 0 - i; offset <= RADIUS; offset++)
            {
                result += vec[i + offset];
            }
        }

        if (i < size && i >= size - RADIUS)
        {
            for (int offset = -RADIUS; offset < size - i; offset++)
            {
                result += vec[i + offset];
            }
        }

        result_vec[i] = result;
    }
}

// Compare result vectors
int compareResultVec(int *vectorCPU, int *vectorGPU, int size)
{
    int error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(vectorCPU[i] - vectorGPU[i]);
    }
    if (error == 0)
    {
        cout << "No errors. All good!" << endl;
        return 0;
    }
    else
    {
        cout << "Accumulated error: " << error << endl;
        return -1;
    }
}

int main(void)
{
    // Allocate and prepare input vector on host memory
    int *hostVecInput = new int[WIDTH];
    int *hostVecOutputCPU = new int[WIDTH];
    int *hostVecOutputGPU1 = new int[WIDTH];
    int *hostVecOutputGPU2 = new int[WIDTH];
    int *hostVecOutputGPU3 = new int[WIDTH];
    int *hostVecOutputGPU4 = new int[WIDTH];

    for (int i = 0; i < WIDTH; i++)
    {
        hostVecInput[i] = 1;
    }

    // Get the CPU result
    movingSumCPU(hostVecInput, hostVecOutputCPU, WIDTH);

    // Allocate device memory
    int *deviceVecInput;
    int *deviceVecOutput1;
    int *deviceVecOutput2;
    int *deviceVecOutput3;
    int *deviceVecOutput4;
    gpuErrCheck(cudaMalloc(&deviceVecInput, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput1, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput2, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput3, WIDTH * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVecOutput4, WIDTH * sizeof(int)));

    // Copy data from host to device
    gpuErrCheck(cudaMemcpy(deviceVecInput, hostVecInput, WIDTH * sizeof(int), cudaMemcpyHostToDevice));

    // Run kernel on all elements on the GPU
    int nbr_blocks = ((WIDTH % BLOCKSIZE) != 0) ? (WIDTH / BLOCKSIZE + 1) : (WIDTH / BLOCKSIZE);
    movingSumGlobal<<<nbr_blocks, BLOCKSIZE>>>(deviceVecInput, deviceVecOutput1, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());
    movingSumSharedMemStatic<<<nbr_blocks, BLOCKSIZE>>>(deviceVecInput, deviceVecOutput2, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());
    int bytesInSharedMemory = (BLOCKSIZE + 2 * RADIUS) * sizeof(int);
    movingSumSharedMemDynamic<<<nbr_blocks, BLOCKSIZE, bytesInSharedMemory>>>(deviceVecInput, deviceVecOutput3, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());
    movingSumAtomics<<<nbr_blocks, BLOCKSIZE>>>(deviceVecInput, deviceVecOutput4, WIDTH);
    gpuErrCheck(cudaPeekAtLastError());

    // Copy the result stored in device_y back to host_y
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU1, deviceVecOutput1, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU2, deviceVecOutput2, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU3, deviceVecOutput3, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrCheck(cudaMemcpy(hostVecOutputGPU4, deviceVecOutput4, WIDTH * sizeof(int), cudaMemcpyDeviceToHost));

    // Check for errors in result
    auto ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU1, WIDTH);
    ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU2, WIDTH);
    ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU3, WIDTH);
    ret = compareResultVec(hostVecOutputCPU, hostVecOutputGPU4, WIDTH);

    // Free memory on device & host
    cudaFree(deviceVecInput);
    cudaFree(deviceVecOutput1);
    cudaFree(deviceVecOutput2);
    cudaFree(deviceVecOutput3);
    cudaFree(deviceVecOutput4);
    delete[] hostVecInput;
    delete[] hostVecOutputCPU;
    delete[] hostVecOutputGPU1;
    delete[] hostVecOutputGPU2;
    delete[] hostVecOutputGPU3;
    delete[] hostVecOutputGPU4;

    return 0;
}