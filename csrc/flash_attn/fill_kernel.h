#pragma once

#include <cuda_runtime.h>

#include <cstddef>

void fill_float(float *data, float value, size_t num_elements, cudaStream_t stream);
