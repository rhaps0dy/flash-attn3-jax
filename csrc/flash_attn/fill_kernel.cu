#include "fill_kernel.h"
#include <cuda_runtime.h>
#include <limits>

__global__ void fill_float_kernel(float *data, const float value, const size_t size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = value;
  }
}

void fill_float(float *data, float value, size_t num_elements, cudaStream_t stream) {
  if (num_elements > 0) {
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    fill_float_kernel<<<grid_size, block_size, 0, stream>>>(
        data, value, num_elements);
  }
}
