#pragma once

// rocSHMEM extension header for PyTorch distributed symmetric memory.
//
// Declares the c10d::rocshmem_extension namespace with ROCm-specific
// functions that are exposed as separate Python bindings in init.cpp
// (e.g. _rocshmem_hipmodule_init, _is_rocshmem_available).
//
// The collective operations (put, get, broadcast, all_to_all, …) are
// implemented under c10d::nvshmem_extension in rocshmem_extension.cu so
// that the existing TORCH_LIBRARY_IMPL registrations and init.cpp bindings
// for _nvshmemx_cumodule_init / _is_nvshmem_available continue to work
// unchanged on both CUDA (nvshmem_extension.cu) and ROCm (rocshmem_extension.cu).

#if defined(USE_ROCM)

#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.hpp>

namespace c10d::rocshmem_extension {

// Returns true when rocSHMEM is compiled in (always true for this library).
TORCH_API bool is_rocshmem_available();

// Initializes the device state in hipModule_t so that it can perform
// rocSHMEM device-side operations. Wraps rocshmem_hipmodule_init().
TORCH_API void rocshmem_hipmodule_init(uintptr_t module);

} // namespace c10d::rocshmem_extension

#endif  // USE_ROCM
