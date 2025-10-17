import os

# Ring attention tests require the latency-hiding scheduler to overlap
# collective-permute operations with custom-call kernels. This must be set
# before JAX initializes, so we set it here in conftest.py which pytest
# loads before any test module.
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_latency_hiding_scheduler=true"
