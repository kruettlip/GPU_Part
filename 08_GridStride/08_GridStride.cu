/* This program will do a vector addition on two vecotrs.
 *  They have the same size N (defined in main).
 *
 *  We will use the concept of the GridStride loop to iterate
 *  over the data on the GPU kernel.
 *
 *  +-------+   +-------+   +-------+
 *  |1|2|3|4| + |1|2|3|4| = |2|4|6|8|
 *  +-------+   +-------+   +-------+
 *
 * The next step is to extend this concept to a 2D data layout.
 * The GridStride loop has to iterate over the 2D data and still
 * have coalesced memory access and the same flexibility.
 *
 *  +-------+   +-------+   +-------+
 *  |1|2|3|4| + |1|2|3|4| = |2|4|6|8|
 *  |1|2|3|4| + |1|2|3|4| = |2|4|6|8|
 *  |1|2|3|4| + |1|2|3|4| = |2|4|6|8|
 *  +-------+   +-------+   +-------+
 *
 */
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

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

// CPU kernel function to add the elements of two arrays (called vectors)
void add(int *vectorA, int *vectorB, int *vectorC, int size)
{
    // PS: You know how to do this in AVX2, don't you?
    for (int i = 0; i < size; i++)
        vectorC[i] = vectorA[i] + vectorB[i];
}

// Kernel function to add the elements of two arrays
__global__ void cudaAddMonolithic(int *vectorA, int *vectorB, int *vectorC, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        vectorC[idx] = vectorA[idx] + vectorB[idx];
    }
}

// Kernel function to add the elements of two arrays using 1D GridStride loop
__global__ void cudaAdd1DGridStride(int *vectorA, int *vectorB, int *vectorC, int size)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x)
    {
        vectorC[idx] = vectorA[idx] + vectorB[idx];
    }
}

// Kernel function to add the elements of two arrays in 2D
__global__ void cudaAdd2DMonolithic(int *matrixA, int *matrixB, int *matrixC, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height)
    {
        int globalIdx = idy * width + idx;
        matrixC[globalIdx] = matrixA[globalIdx] + matrixB[globalIdx];
    }
}

// Kernel function to add the elements of two arrays using 2D GridStride loop
__global__ void cudaAdd2DGridStride(int *matrixA, int *matrixB, int *matrixC, int width, int height)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < width; idx += blockDim.x * gridDim.x)
    {
        for (int idy = blockIdx.y * blockDim.y + threadIdx.y; idy < height; idy += blockDim.y * gridDim.y)
        {
            int globalIdx = idy * width + idx;
            matrixC[globalIdx] = matrixA[globalIdx] + matrixB[globalIdx];
        }
    }
}

// Compare result arrays CPU vs GPU result. If no diff, the result pass.
int compareResultVec(int *vectorCPU, int *vectorGPU, int size, string name)
{
    int error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(vectorCPU[i] - vectorGPU[i]);
    }
    if (error == 0)
    {
        cout << name << ": Test passed." << endl;
        return 0;
    }
    else
    {
        cout << name << ": Accumulated error: " << error << endl;
        return -1;
    }
}

int main(void)
{
    // Define the size of the vector: 1048576 elements
    int N = 1 << 20;

    // Allocate and prepare input/output arrays on host memory
    int *hostVectorA = new int[N];
    int *hostVectorB = new int[N];
    int *hostVectorCCPU = new int[N];
    int *hostVectorCGPU = new int[N];
    for (int i = 0; i < N; i++)
    {
        hostVectorA[i] = i;
        hostVectorB[i] = i;
    }

    // Alloc N times size of int at device memory for deviceVector[A-C]
    int *deviceVectorA;
    int *deviceVectorB;
    int *deviceVectorC;
    gpuErrCheck(cudaMalloc(&deviceVectorA, N * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVectorB, N * sizeof(int)));
    gpuErrCheck(cudaMalloc(&deviceVectorC, N * sizeof(int)));

    // Copy data from host to device
    gpuErrCheck(cudaMemcpy(deviceVectorA, hostVectorA, N * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(deviceVectorB, hostVectorB, N * sizeof(int), cudaMemcpyHostToDevice));

    // Run the vector kernel on the CPU
    add(hostVectorA, hostVectorB, hostVectorCCPU, N);

    // 1 thread per data point
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAddMonolithic<<<1024, 1024>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "Monolithic                ");

    // Serial execution
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd1DGridStride<<<1, 1>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride <<<1,1>>>      ");

    // Use some random number of threads
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd1DGridStride<<<1, 42>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride <<<1,42>>>     ");

    // Only use 1 block
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd1DGridStride<<<1, 1024>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride <<<1,1024>>>   ");

    // Use 32 blocks
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd1DGridStride<<<32, 1024>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride <<<32,1024>>>  ");

    // Use 1024 blocks --> 1 thread per data point
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd1DGridStride<<<1024, 1024>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride <<<1024,1024>>>");

    // Use too much blocks
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd1DGridStride<<<2048, 1024>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride <<<2048,1024>>>");

    // Try to balance num blocks per SM
    int numSMs;
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0); // hard coded device 0
    cudaAdd1DGridStride<<<8 * numSMs, 1024>>>(deviceVectorA, deviceVectorB, deviceVectorC, N);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    string txt = "GridStride <<<8*" + to_string(numSMs) + ",1024>>>";
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, txt);

    // -------------------- Use the same buffer but access it in 2D by 1024x1024 --------------------
    // 1 thread per data point
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DMonolithic<<<dim3(32, 32), dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 1024, 1024);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "Monolithic2D (1024x1024)                     ");

    // Serial execution
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<1, 1>>>(deviceVectorA, deviceVectorB, deviceVectorC, 1024, 1024);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (1024x1024)<<<1,1>>>            ");

    // Only use 1 block
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<1, dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 1024, 1024);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (1024x1024)<<<1,(32,32)>>>      ");

    // Use 16 blocks
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<dim3(4, 4), dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 1024, 1024);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (1024x1024)<<<(4,4),(32,32)>>>  ");

    // Use 1024 blocks --> 1 thread per data point
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<dim3(32, 32), dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 1024, 1024);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (1024x1024)<<<(32,32),(32,32)>>>");

    // Use too much blocks
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<dim3(48, 48), dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 1024, 1024);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (1024x1024)<<<(48,48),(32,32)>>>");

    // Try to balance num blocks per SM
    int nbrBlocks = ceil(sqrt(numSMs * 8));
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<dim3(nbrBlocks, nbrBlocks), dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 1024, 1024);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    txt = "GridStride2D (1024x1024)<<<(" + to_string(nbrBlocks) + "," + to_string(nbrBlocks) + "),(32,32)>>>";
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, txt);

    // -------------------- Use the same buffer but access it in 2D by 512x2048 --------------------
    // 1 thread per data point
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DMonolithic<<<dim3(16, 64), dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 512, 2048);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "Monolithic2D (512x2048)                     ");

    // Serial execution
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<1, 1>>>(deviceVectorA, deviceVectorB, deviceVectorC, 512, 2048);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (512x2048)<<<1,1>>>            ");

    // Only use 1 block
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<1, dim3(16, 64)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 512, 2048);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (512x2048)<<<1,(16,64)>>>      ");

    // Use 16 blocks with 32x32 threads
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<dim3(4, 4), dim3(32, 32)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 512, 2048);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (512x2048)<<<(4,4),(32,32)>>>  ");

    // Use 1024 blocks --> 1 thread per data point
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<dim3(16, 64), dim3(16, 64)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 512, 2048);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (512x2048)<<<(16,64),(16,64)>>>");

    // Use too much blocks and don't proper map the 2D layout
    gpuErrCheck(cudaMemset(deviceVectorC, 0, N));
    cudaAdd2DGridStride<<<dim3(48, 48), dim3(16, 64)>>>(deviceVectorA, deviceVectorB, deviceVectorC, 512, 2048);
    gpuErrCheck(cudaPeekAtLastError());
    gpuErrCheck(cudaMemcpy(hostVectorCGPU, deviceVectorC, N * sizeof(int), cudaMemcpyDeviceToHost));
    compareResultVec(hostVectorCCPU, hostVectorCGPU, N, "GridStride2D (512x2048)<<<(48,48),(16,64)>>>");

    // Free memory on device
    gpuErrCheck(cudaFree(deviceVectorA));
    gpuErrCheck(cudaFree(deviceVectorB));
    gpuErrCheck(cudaFree(deviceVectorC));

    // Free memory on host
    delete[] hostVectorA;
    delete[] hostVectorB;
    delete[] hostVectorCCPU;
    delete[] hostVectorCGPU;

    return 0;
}