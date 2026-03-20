"""
rocSHMEM Triton integration for AMD GPU (ROCm) builds.

This module provides the rocSHMEM-specific counterpart to the NVSHMEM Triton
integration in _nvshmem_triton.py.  It is only imported on ROCm builds
(i.e. when ``torch.version.hip is not None``).
"""

import logging
import os
import sysconfig
from typing import Any

import torch
from torch.utils._triton import has_triton


logger = logging.getLogger(__name__)


class RocshmemLibFinder:
    """
    Find the architecture-specific rocSHMEM device bitcode library.

    Environment variable:
        ``ROCSHMEM_LIB_DIR`` (Optional[str]): directory containing
        ``librocshmem_device_{arch}.bc``.  When not set, the standard
        ROCm installation at ``/opt/rocm/lib`` is searched.

    Example::
        export ROCSHMEM_LIB_DIR=/opt/rocm/lib
    """

    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        if cls.found_device_lib_path is not None:
            return cls.found_device_lib_path

        if not torch.cuda.is_available():
            raise RuntimeError(
                "ROCm/CUDA not available — cannot detect GPU architecture"
            )

        props = torch.cuda.get_device_properties(0)
        # gcnArchName returns e.g. "gfx942:sramecc+:xnack-"
        arch = props.gcnArchName.split(":")[0]
        logger.info("Detected GPU architecture: %s", arch)

        lib_name = f"librocshmem_device_{arch}.bc"

        user_lib_dir = os.environ.get("ROCSHMEM_LIB_DIR")
        if user_lib_dir is not None:
            lib_path = os.path.join(user_lib_dir, lib_name)
            if not os.path.exists(lib_path):
                raise RuntimeError(
                    f"rocSHMEM device library not found at ROCSHMEM_LIB_DIR: "
                    f"{lib_path}"
                )
            cls.found_device_lib_path = lib_path
            return lib_path

        search_paths = [
            os.path.join(sysconfig.get_path("purelib"), "amd", "rocshmem", "lib"),
            "/opt/rocm/lib",
            "/opt/rocm-7.1.0/lib",
            "/usr/local/lib",
            "/usr/lib",
        ]

        for path in search_paths:
            candidate = os.path.join(path, lib_name)
            if os.path.exists(candidate):
                logger.info("Found rocSHMEM device library: %s", candidate)
                cls.found_device_lib_path = candidate
                return candidate

        raise RuntimeError(
            f"rocSHMEM device library '{lib_name}' not found.\n"
            f"Searched: {search_paths}\n"
            f"Set ROCSHMEM_LIB_DIR to the directory containing it."
        )


class RocshmemKernelRegistry:
    """Track Triton kernels that need rocSHMEM HIP-module initialization."""

    _to_init: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> None:
        cls._to_init.setdefault(name)

    @classmethod
    def deregister(cls, name: str) -> None:
        cls._to_init.pop(name, None)

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._to_init


def _rocshmem_init_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """
    Post-compile hook that initializes rocSHMEM device context in the
    compiled HIP module.  Mirrors ``_nvshmem_init_hook`` but calls
    ``_rocshmem_hipmodule_init`` instead of ``_nvshmemx_cumodule_init``.
    """

    try:
        # Try the new canonical name first; fall back to the old name for
        # builds where libtorch_python.so predates the rename.
        try:
            from torch._C._distributed_c10d import _rocshmem_hipmodule_init
        except ImportError:
            from torch._C._distributed_c10d import (  # type: ignore[no-redef]
                _rocshmemx_hipmodule_init as _rocshmem_hipmodule_init,
            )
    except ImportError as e:
        raise RuntimeError(
            "rocSHMEM C++ extension not found. "
            "PyTorch must be built with USE_ROCM=1 and rocSHMEM installed."
        ) from e

    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__

    if not RocshmemKernelRegistry.has(fn_name):
        return

    key = kwargs["key"]
    device = kwargs["compile"]["device"]
    kernel_cache = jit_function.device_caches[device][0]
    kernel = kernel_cache.get(key, None)
    if kernel is not None:
        kernel.run  # noqa: B018 — touch the JIT cache entry
        _rocshmem_hipmodule_init(kernel.module)
    else:
        logger.warning(
            "It seems Triton hasn't created a kernel for function %s. "
            "Please report this issue to Triton.",
            fn_name,
        )


if has_triton():
    import triton
    import triton.language as tl
    from triton.language import core
    from triton.runtime.jit import JITFunction, KernelInterface

    class GridCallableWithExtern(KernelInterface):
        """
        ``KernelInterface`` invokes ``self.run`` in ``__getitem__``, i.e. [].
        We implement a ``run`` method by directing the call to
        ``JITFunction.run``, with added ``extern_libs`` kwarg, so that users
        don't have to pass it.
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)

    def requires_rocshmem(  # type: ignore[no-untyped-def]
        jit_func,
    ):
        """
        Decorator to mark a Triton kernel as requiring rocSHMEM device APIs.

        Finds the architecture-specific rocSHMEM bitcode library, registers
        the kernel for post-compile HIP-module initialization, and wraps the
        function so that ``extern_libs`` is injected automatically.

        Example::

            @requires_rocshmem
            @triton.jit
            def my_kernel(...):
                pe = rocshmem_my_pe()
                rocshmem_putmem_wg(dest, src, nbytes, target_pe)

        Set ``ROCSHMEM_LIB_DIR`` to override the default library search path.
        """
        if not isinstance(jit_func, JITFunction):
            raise TypeError(
                f"@requires_rocshmem must be applied to a @triton.jit function, "
                f"got {type(jit_func)}"
            )

        lib_path = RocshmemLibFinder.find_device_library()
        # Key must be a substring of the rocSHMEM device function names
        # (e.g. "rocshmem_my_pe") so that amd.need_extern_lib() returns True
        # and the bitcode is actually linked into the Triton kernel.
        extern_libs = {"rocshmem": lib_path}

        RocshmemKernelRegistry.register(jit_func.fn.__name__)
        triton.knobs.runtime.jit_post_compile_hook = _rocshmem_init_hook

        return GridCallableWithExtern(jit_func, extern_libs)

    # -----------------------------------------------------------------------
    # rocSHMEM device API — Triton-callable device functions.
    #
    # All symbol names are the direct rocSHMEM device-bitcode names; no
    # NVSHMEM → rocSHMEM translation layer is needed here.
    #
    # RMA (block/wg-scoped):
    #   nvshmemx_putmem_block  → rocshmem_putmem_wg
    #   nvshmemx_getmem_block  → rocshmem_getmem_wg
    #   nvshmemx_getmem_nbi_block → rocshmem_getmem_nbi_wg
    #   nvshmemx_putmem_signal_block → rocshmem_putmem_signal_wg
    # Wait / signal:
    #   nvshmem_int_wait_until    → rocshmem_int_wait_until
    #   nvshmem_signal_wait_until → rocshmem_uint64_wait_until
    # Memory ordering, PE info, barriers: names are identical.
    # -----------------------------------------------------------------------

    @triton.jit
    def put(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """Put *nelems* elements from local *source* to *dest* on remote *pe*."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _putmem_wg(dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe)

    @core.extern
    def _putmem_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "", "",
            [dest, source, size_bytes, pe],
            {(core.dtype("int64"), core.dtype("int64"), core.dtype("int64"), core.dtype("int32")):
             ("rocshmem_putmem_wg", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @triton.jit
    def get(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """Get *nelems* elements from *source* on remote *pe* into local *dest* (blocking)."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _getmem_wg(dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe)

    @core.extern
    def _getmem_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "", "",
            [dest, source, size_bytes, pe],
            {(core.dtype("int64"), core.dtype("int64"), core.dtype("int64"), core.dtype("int32")):
             ("rocshmem_getmem_wg", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @triton.jit
    def get_nbi(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """Non-blocking get; call quiet() for completion."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _getmem_nbi_wg(dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe)

    @core.extern
    def _getmem_nbi_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "", "",
            [dest, source, size_bytes, pe],
            {(core.dtype("int64"), core.dtype("int64"), core.dtype("int64"), core.dtype("int32")):
             ("rocshmem_getmem_nbi_wg", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @triton.jit
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst, src, size_bytes, signal, sig_val, sig_op, pe,
    ):
        """Put data to remote PE and atomically update a signal variable."""
        sig_val = 0 << 32 | sig_val
        return _putmem_signal_wg(
            dst.to(tl.int64), src.to(tl.int64), size_bytes.to(tl.int64),
            signal.to(tl.int64), sig_val.to(tl.uint64), sig_op, pe,
        )

    @core.extern
    def _putmem_signal_wg(  # type: ignore[no-untyped-def]
        dst, src, size_bytes, signal, sig_val, sig_op, pe, _semantic=None,
    ):
        return core.extern_elementwise(
            "", "",
            [dst, src, size_bytes, signal, sig_val, sig_op, pe],
            {(core.dtype("int64"), core.dtype("int64"), core.dtype("int64"),
              core.dtype("int64"), core.dtype("uint64"), core.dtype("int32"), core.dtype("int32")):
             ("rocshmem_putmem_signal_wg", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @triton.jit
    def wait_until(ivar, cmp_op, cmp_val):  # type: ignore[no-untyped-def]
        """Block until *ivar* satisfies the comparison condition."""
        tl.static_assert(
            ivar.type.element_ty.itemsize == 4,
            "wait_until expects a 32-bit type for the synchronization variable",
        )
        return _int_wait_until(ivar.to(tl.int64), cmp_op, cmp_val)

    @core.extern
    def _int_wait_until(ivar, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "", "",
            [ivar, cmp, cmp_val],
            {(core.dtype("int64"), core.dtype("int32"), core.dtype("int32")):
             ("rocshmem_int_wait_until", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @triton.jit
    def signal_wait_until(signal, cmp, cmp_val):  # type: ignore[no-untyped-def]
        """Block until a uint64 signal variable satisfies the comparison condition."""
        cmp_val = 0 << 32 | cmp_val
        return _uint64_wait_until(signal.to(tl.int64), cmp, cmp_val.to(tl.uint64))

    @core.extern
    def _uint64_wait_until(signal, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "", "",
            [signal, cmp, cmp_val],
            {(core.dtype("int64"), core.dtype("int32"), core.dtype("uint64")):
             ("rocshmem_uint64_wait_until", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @triton.jit
    def signal_op(sig_addr, signal, sig_op, pe):  # type: ignore[no-untyped-def]
        """Not available in rocSHMEM device bitcode."""
        tl.static_assert(
            False,
            "rocshmem has no device-bitcode equivalent for signal_op. "
            "Use rocshmem_uint64_atomic_set or rocshmem_uint64_atomic_add instead.",
        )

    @core.extern
    def fence(_semantic=None):  # type: ignore[no-untyped-def]
        """Ensure ordering of put operations to each remote PE."""
        return core.extern_elementwise(
            "", "", [],
            {(): ("rocshmem_fence", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @core.extern
    def quiet(_semantic=None):  # type: ignore[no-untyped-def]
        """Wait for completion of all outstanding put operations."""
        return core.extern_elementwise(
            "", "", [],
            {(): ("rocshmem_quiet", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @core.extern
    def my_pe(_semantic=None):  # type: ignore[no-untyped-def]
        """Return the PE number of the calling PE."""
        return core.extern_elementwise(
            "", "", [],
            {(): ("rocshmem_my_pe", core.dtype("int32"))},
            is_pure=True, _semantic=_semantic,
        )

    @core.extern
    def n_pes(_semantic=None):  # type: ignore[no-untyped-def]
        """Return the total number of PEs."""
        return core.extern_elementwise(
            "", "", [],
            {(): ("rocshmem_n_pes", core.dtype("int32"))},
            is_pure=True, _semantic=_semantic,
        )

    @core.extern
    def barrier_all(_semantic=None):  # type: ignore[no-untyped-def]
        """Barrier across all PEs with completion guarantee."""
        return core.extern_elementwise(
            "", "", [],
            {(): ("rocshmem_barrier_all", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    @core.extern
    def sync_all(_semantic=None):  # type: ignore[no-untyped-def]
        """Lightweight synchronization barrier across all PEs."""
        return core.extern_elementwise(
            "", "", [],
            {(): ("rocshmem_sync_all", core.dtype("int32"))},
            is_pure=False, _semantic=_semantic,
        )

    # Collective stubs — wg-scoped variants not in rocSHMEM device bitcode yet.

    @triton.jit
    def alltoall(team, dest, source, nelems_per_pe):  # type: ignore[no-untyped-def]
        """Not available: rocshmem_alltoallmem_wg is not in the device bitcode."""
        tl.static_assert(
            False,
            "rocshmem_alltoallmem_wg is not available in the device bitcode. "
            "Use host-side rocshmem_alltoallmem_on_stream instead.",
        )

    @triton.jit
    def broadcast(team, dest, source, nelems, pe_root):  # type: ignore[no-untyped-def]
        """Not available: rocshmem_broadcastmem_wg is not in the device bitcode."""
        tl.static_assert(
            False,
            "rocshmem_broadcastmem_wg is not available in the device bitcode. "
            "Use host-side rocshmem_broadcastmem_on_stream instead.",
        )

    @triton.jit
    def reduce(team, dest, source, nreduce, operation: tl.constexpr):  # type: ignore[no-untyped-def]
        """Not available: rocshmem team reduce wg ops are not in the device bitcode."""
        tl.static_assert(
            False,
            "rocshmem team reduce is not available in the device bitcode. "
            "Use host-side rocshmem reduce API instead.",
        )
