# Owner(s): ["oncall: distributed"]
# To run:
# python test/distributed/test_nvshmem_triton.py
import csv
import os
import sys
import unittest

import torch
import torch.distributed._symmetric_memory as symm_mem

# Skip entire module on ROCm before importing NVSHMEM-specific modules
if not symm_mem.is_nvshmem_available():
    print("SHMEM backend (NVSHMEM/rocSHMEM) not available, skipping tests")
    sys.exit(0)

# Shared Triton JIT kernels

import triton.language as tl

import torch.distributed as dist
import torch.distributed._symmetric_memory._shmem_triton as shmem_triton
from torch._inductor.runtime.triton_compat import triton
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)
from torch.testing._internal.inductor_utils import IS_H100, requires_triton

try:
    from tritonblas.kernels.stages import GemmContext, ScheduleContext, make_tensor_view

    HAS_TRITONBLAS = True
except ImportError:
    GemmContext = None
    ScheduleContext = None
    make_tensor_view = None
    HAS_TRITONBLAS = False

try:
    import tritonblas as _tritonblas_mod

    HAS_TRITONBLAS_MATMUL = True
except ImportError:
    _tritonblas_mod = None
    HAS_TRITONBLAS_MATMUL = False


shmem_backend = shmem_triton.get_shmem_backend_module()
requires_shmem = shmem_triton.requires_shmem

ROCM_COLLECTIVE_UNAVAILABLE_REASON = (
    "rocSHMEM *_wg collective symbols are unavailable in current device bitcode for this op."
)
ROCM_PUT_SIGNAL_ADD_HANG_REASON = (
    "Known hang in rocSHMEM Triton put_signal_add path."
)


device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_shmem
@triton.jit
def my_put_kernel(dest, src, nelems, pe):
    shmem_backend.put(dest, src, nelems, pe)


@requires_shmem
@triton.jit
def my_get_kernel(dest, src, nelems, pe, nbi: tl.constexpr):
    if nbi:
        shmem_backend.get_nbi(dest, src, nelems, pe)
        shmem_backend.quiet()
    else:
        shmem_backend.get(dest, src, nelems, pe)


@requires_shmem
@triton.jit
def my_putmem_signal_block_kernel(
    dst,
    src,
    size_bytes,
    signal,
    sig_val,
    sig_op,
    peer,
):
    shmem_backend.putmem_signal_block(
        dst, src, size_bytes, signal, sig_val, sig_op, peer
    )


@requires_shmem
@triton.jit
def my_signal_wait_until_kernel(signal, cmp_op, cmp_val):
    shmem_backend.signal_wait_until(signal, cmp_op, cmp_val)


@requires_shmem
@triton.jit
def my_signal_op_kernel(
    sig_addr,
    signal,
    sig_op,
    peer,
):
    shmem_backend.signal_op(sig_addr, signal, sig_op, peer)


@requires_shmem
@triton.jit
def my_wait_until_kernel(ivar, cmp_op, cmp_val):
    shmem_backend.wait_until(ivar, cmp_op, cmp_val)


@requires_shmem
@triton.jit
def my_fence_kernel():
    shmem_backend.fence()


@requires_shmem
@triton.jit
def my_put_with_fence_kernel(
    dst1,
    src1,
    dst2,
    src2,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    # First put
    shmem_backend.put(dst1, src1, nelems, peer)
    # Ensure the first put is ordered before the next.
    shmem_backend.fence()
    # Second put
    shmem_backend.put(dst2, src2, nelems, peer)
    # Order the second put before flag update.
    shmem_backend.fence()
    # Write the flag (single int64) to signal completion.
    shmem_backend.put(flag_dst, flag_src, 1, peer)


@requires_shmem
@triton.jit
def my_put_with_quiet_kernel(
    dst,
    src,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    # Put data
    shmem_backend.put(dst, src, nelems, peer)
    # Call quiet to ensure put is complete
    shmem_backend.quiet()
    # Only after quiet, set the completion flag
    # This ensures the data put is complete before flag is set
    shmem_backend.put(flag_dst, flag_src, 1, peer)


@requires_shmem
@triton.jit
def my_barrier_test_kernel(dst, src, nelems):
    # Testing barrier_all() requires coordinated operations across PEs within
    # the same kernel execution. Unlike other kernels that just wrap NVSHMEM
    # primitives, this one implements the full test logic to properly verify
    # device-side barrier synchronization.
    my_pe = shmem_backend.my_pe()
    n_pes = shmem_backend.n_pes()

    # Rank 0 broadcasts its value to all other ranks
    if my_pe == 0:
        # Write initial value
        p_src = src.to(tl.pointer_type(tl.int32))
        tl.store(p_src, 42)
        # Put to all other ranks
        i = 1
        while i < n_pes:
            shmem_backend.put(dst, src, nelems, i)
            i += 1

    # Synchronize all PEs
    shmem_backend.barrier_all()

    # Non-zero ranks increment the received value
    if my_pe != 0:
        p_dst = dst.to(tl.pointer_type(tl.int32))
        received = tl.load(p_dst)
        tl.store(p_dst, received + 1)


@requires_shmem
@triton.jit
def my_sync_test_kernel(local_data, remote_data, nelems):
    my_pe = shmem_backend.my_pe()
    n_pes = shmem_backend.n_pes()

    # Each PE writes a unique value to its local memory
    p_local = local_data.to(tl.pointer_type(tl.int32))
    unique_value = my_pe + 100
    tl.store(p_local, unique_value)

    # sync_all() ensures local stores are visible to other PEs
    # but doesn't guarantee completion of any remote operations
    shmem_backend.sync_all()

    # Now each PE reads from the next PE's memory to verify visibility
    # PE 0 reads from PE 1, PE 1 reads from PE 2, ..., PE n-1 reads from PE 0
    next_pe = (my_pe + 1) % n_pes
    shmem_backend.get(remote_data, local_data, nelems, next_pe)

    # The get should now see the value that the next PE wrote locally
    # because sync_all() made those local stores visible


@requires_shmem
@triton.jit
def my_barrier_all_kernel():
    shmem_backend.barrier_all()


@requires_shmem
@triton.jit
def my_alltoall_kernel(
    team_handle,
    dst,
    src,
    nelems_per_pe,
):
    shmem_backend.alltoall(team_handle, dst, src, nelems_per_pe)


@requires_shmem
@triton.jit
def my_broadcast_kernel(
    team_handle,
    dst,
    src,
    nelems,
    pe_root,
):
    shmem_backend.broadcast(team_handle, dst, src, nelems, pe_root)


@requires_shmem
@triton.jit
def my_reduce_kernel(
    team_handle,
    dest_tensor,
    source_tensor,
    nreduce,
    operation: tl.constexpr,
):
    shmem_backend.reduce(team_handle, dest_tensor, source_tensor, nreduce, operation)


@requires_shmem
@triton.jit
def my_fused_get_add_kernel(
    out,
    local_inp,
    remote_symm_src,
    tmp_remote,
    nelems,
    peer,
    block_size: tl.constexpr,
):
    # Pull remote data with SHMEM and fuse local compute in one kernel launch.
    shmem_backend.get(tmp_remote, remote_symm_src, nelems, peer)
    offsets = tl.arange(0, block_size)
    mask = offsets < nelems
    lhs = tl.load(local_inp + offsets, mask=mask)
    rhs = tl.load(tmp_remote + offsets, mask=mask)
    tl.store(out + offsets, lhs + rhs, mask=mask)


@requires_shmem
@triton.jit
def my_put_quiet_kernel(dest, src, nelems, pe):
    shmem_backend.put(dest, src, nelems, pe)
    shmem_backend.quiet()


@requires_shmem
@triton.jit
def my_parallel_put_quiet_kernel(
    src_ptr,
    total_elems,
    peer,
    chunk_per_wg: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * chunk_per_wg
    if start < total_elems:
        remaining = total_elems - start
        count = chunk_per_wg
        if remaining < chunk_per_wg:
            count = remaining
        shmem_backend.put(src_ptr + start, src_ptr + start, count, peer)
    shmem_backend.quiet()


@requires_shmem
@triton.jit
def my_parallel_put_allpeers_quiet_kernel(
    src_ptr,
    total_elems,
    rank,
    chunk_per_wg: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * chunk_per_wg
    if start < total_elems:
        remaining = total_elems - start
        count = chunk_per_wg
        if remaining < chunk_per_wg:
            count = remaining
        for p in tl.static_range(0, world_size):
            if p != rank:
                shmem_backend.put(src_ptr + start, src_ptr + start, count, p)
    shmem_backend.quiet()


@requires_shmem
@triton.jit
def my_parallel_put_no_quiet_kernel(
    src_ptr,
    total_elems,
    peer,
    chunk_per_wg: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * chunk_per_wg
    if start < total_elems:
        remaining = total_elems - start
        count = chunk_per_wg
        if remaining < chunk_per_wg:
            count = remaining
        shmem_backend.put(src_ptr + start, src_ptr + start, count, peer)


@requires_shmem
@triton.jit
def my_parallel_put_striped_peers_quiet_kernel(
    src_ptr,
    total_elems,
    rank,
    chunk_per_wg_per_peer: tl.constexpr,
    world_size: tl.constexpr,
    wgs_per_peer: tl.constexpr,
):
    """Each WG handles exactly 1 peer, 1 chunk. Grid = wgs_per_peer * (world_size-1)."""
    pid = tl.program_id(0)
    num_peers = world_size - 1
    peer_idx = pid % num_peers
    wg_within_peer = pid // num_peers

    peer = peer_idx
    if peer >= rank:
        peer = peer + 1

    start = wg_within_peer * chunk_per_wg_per_peer
    if start < total_elems:
        remaining = total_elems - start
        count = chunk_per_wg_per_peer
        if remaining < chunk_per_wg_per_peer:
            count = remaining
        shmem_backend.put(src_ptr + start, src_ptr + start, count, peer)
    shmem_backend.quiet()


@requires_shmem
@triton.jit
def my_quiet_kernel():
    shmem_backend.quiet()


@requires_shmem
@triton.jit
def my_fused_matmul_allgather_put_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_local,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    rank,
    peer,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    k_iter = 0
    while k_iter < k:
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k[None, :] + k_iter) * stride_ak
        b_ptrs = b_ptr + (offs_k[:, None] + k_iter) * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < m_local) & ((offs_k[None, :] + k_iter) < k)
        b_mask = ((offs_k[:, None] + k_iter) < k) & (offs_n[None, :] < n)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        k_iter += block_k

    c = acc.to(tl.float16)
    global_rows = offs_m + rank * m_local
    c_ptrs = c_ptr + global_rows[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m_local) & (offs_n[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)

    # Ship this tile to peer: row-wise contiguous puts for current N-tile.
    col_start = pid_n * block_n
    valid_cols = tl.minimum(block_n, n - col_start)
    row_base_local = pid_m * block_m
    for i in range(block_m):
        row_local = row_base_local + i
        if row_local < m_local:
            row_global = row_local + rank * m_local
            row_ptr = c_ptr + row_global * stride_cm + col_start * stride_cn
            shmem_backend.put(row_ptr, row_ptr, valid_cols, peer)


@requires_shmem
@triton.jit
def my_fused_matmul_allgather_put_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_local,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    rank,
    peer,
    num_pid_n,
    total_tiles,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_id = pid
    while tile_id < total_tiles:
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        k_iter = 0
        while k_iter < k:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k[None, :] + k_iter) * stride_ak
            b_ptrs = b_ptr + (offs_k[:, None] + k_iter) * stride_bk + offs_n[None, :] * stride_bn
            a_mask = (offs_m[:, None] < m_local) & ((offs_k[None, :] + k_iter) < k)
            b_mask = ((offs_k[:, None] + k_iter) < k) & (offs_n[None, :] < n)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)
            k_iter += block_k

        c = acc.to(tl.float16)
        global_rows = offs_m + rank * m_local
        c_ptrs = c_ptr + global_rows[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < m_local) & (offs_n[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)

        col_start = pid_n * block_n
        valid_cols = tl.minimum(block_n, n - col_start)
        row_base_local = pid_m * block_m
        for i in range(block_m):
            row_local = row_base_local + i
            if row_local < m_local:
                row_global = row_local + rank * m_local
                row_ptr = c_ptr + row_global * stride_cm + col_start * stride_cn
                shmem_backend.put(row_ptr, row_ptr, valid_cols, peer)

        tile_id += tl.num_programs(0)


@requires_shmem
@triton.jit
def my_fused_matmul_allgather_put_tritonblas_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_local,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    rank,
    peer,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_sms: tl.constexpr,
    group_size_m: tl.constexpr,
    even_k: tl.constexpr,
    even_n: tl.constexpr,
):
    tensor_a = make_tensor_view(a_ptr, m_local, k, stride_am, stride_ak)
    tensor_b = make_tensor_view(b_ptr, k, n, stride_bk, stride_bn)

    gemm_ctx = GemmContext(
        block_m,
        block_n,
        block_k,
        num_sms=num_sms,
        num_xcds=1,
        group_size_m=group_size_m,
        even_k=even_k,
        allow_tf32=False,
    )
    sched = ScheduleContext(m_local, n, k, gemm_ctx)

    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        acc = gemm_ctx.reduce_axis(tensor_a, tensor_b, out_tile)
        c = acc.to(tl.float16)

        rm, rn = out_tile.indices()
        global_rm = rm + rank * m_local
        c_ptrs = c_ptr + global_rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < m_local) & (rn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)

        col_start = out_tile.pid_n * block_n
        valid_cols = block_n if even_n else tl.minimum(block_n, n - col_start)
        row_base_local = out_tile.pid_m * block_m
        row_base_global = row_base_local + rank * m_local
        row_ptr_base = c_ptr + row_base_global * stride_cm + col_start * stride_cn
        for i in tl.static_range(0, block_m):
            row_local = row_base_local + i
            if row_local < m_local:
                row_ptr = row_ptr_base + i * stride_cm
                shmem_backend.put(row_ptr, row_ptr, valid_cols, peer)


@triton.jit
def my_tritonblas_gemm_only_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_local,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    rank,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_sms: tl.constexpr,
    group_size_m: tl.constexpr,
    even_k: tl.constexpr,
):
    tensor_a = make_tensor_view(a_ptr, m_local, k, stride_am, stride_ak)
    tensor_b = make_tensor_view(b_ptr, k, n, stride_bk, stride_bn)

    gemm_ctx = GemmContext(
        block_m,
        block_n,
        block_k,
        num_sms=num_sms,
        num_xcds=1,
        group_size_m=group_size_m,
        even_k=even_k,
        allow_tf32=False,
    )
    sched = ScheduleContext(m_local, n, k, gemm_ctx)

    start, total, stride = sched.persistent_tile_range()
    for tile_id in range(start, total, stride):
        out_tile = sched.get_tile_from_idx(tile_id)
        acc = gemm_ctx.reduce_axis(tensor_a, tensor_b, out_tile)
        c = acc.to(tl.float16)

        rm, rn = out_tile.indices()
        global_rm = rm + rank * m_local
        c_ptrs = c_ptr + global_rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < m_local) & (rn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)


@requires_shmem
@triton.jit
def my_fused_matmul_allgather_bulk_tritonblas_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_local,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    rank,
    peer,
    chunk_elems,
    done_counter_ptr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_sms: tl.constexpr,
    group_size_m: tl.constexpr,
    even_k: tl.constexpr,
):
    tensor_a = make_tensor_view(a_ptr, m_local, k, stride_am, stride_ak)
    tensor_b = make_tensor_view(b_ptr, k, n, stride_bk, stride_bn)

    gemm_ctx = GemmContext(
        block_m,
        block_n,
        block_k,
        num_sms=num_sms,
        num_xcds=1,
        group_size_m=group_size_m,
        even_k=even_k,
        allow_tf32=False,
    )
    sched = ScheduleContext(m_local, n, k, gemm_ctx)

    start, total, stride_val = sched.persistent_tile_range()
    for tile_id in range(start, total, stride_val):
        out_tile = sched.get_tile_from_idx(tile_id)
        acc = gemm_ctx.reduce_axis(tensor_a, tensor_b, out_tile)
        c = acc.to(tl.float16)

        rm, rn = out_tile.indices()
        global_rm = rm + rank * m_local
        c_ptrs = c_ptr + global_rm[:, None] * stride_cm + rn[None, :] * stride_cn
        c_mask = (rm[:, None] < m_local) & (rn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)

    # Intra-kernel sync: last SM to finish does bulk put of entire contiguous shard.
    # Atomic release semantics guarantee prior tl.store writes are visible.
    old = tl.atomic_add(done_counter_ptr, 1, sem="release", scope="gpu")
    if old == num_sms - 1:
        local_start = rank * chunk_elems
        src = c_ptr + local_start
        shmem_backend.put(src, src, chunk_elems, peer)
        shmem_backend.quiet()


class ShmemTritonTestBase(MultiProcContinuousTest):
    __test__ = False
    backend_name = "NVSHMEM"

    def setUp(self) -> None:
        super().setUp()
        if self.__class__ is ShmemTritonTestBase:
            self.skipTest("Abstract SHMEM base test class")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend(self.backend_name)

    @requires_triton()
    def test_triton_put(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        # Configuration
        nelems = 5
        dtype = torch.int64
        val = 42 + rank

        # Create symmetric tensors
        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)
        # Fill source tensor with rank-specific pattern
        for i in range(nelems):
            src[i] = val * 10 + i

        # Rendezvous
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Synchronize before operation
        dist.barrier()

        peer = 1 - rank
        if rank == 0:
            # Rank 0 puts its data to Rank 1
            my_put_kernel[(1,)](dst, src, nelems, peer)

        # Synchronize after operation
        dist.barrier()
        if rank == 1:
            # Verify that rank 1 received rank 0's data
            expected = [420 + i for i in range(nelems)]
            torch.testing.assert_close(
                dst, torch.tensor(expected, device=self.device, dtype=dtype)
            )

    def _run_triton_get(self, nbi: bool) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        # Configuration
        numel = 8
        dtype = torch.int8
        val = 7

        # Create symmetric tensors
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        peer = 1 - rank
        if rank == 1:
            # Rank 1 gets data from rank 0 using tensor-aware API
            my_get_kernel[(1,)](out, inp, numel, peer, nbi=nbi)

        if rank == 1:
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    def _run_triton_get_ring(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        world_size = dist.get_world_size()
        # Configuration
        numel = 8
        dtype = torch.int8

        # Each rank fills its input buffer with its own rank value
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        # Ring topology: each rank gets data from the rank to its left
        # rank 0 gets from rank (world_size-1), rank 1 gets from rank 0, etc.
        peer = (rank - 1) % world_size
        # All ranks execute the get operation using tensor-aware API
        my_get_kernel[(1,)](out, inp, numel, peer, nbi=False)

        expected_value = peer
        torch.testing.assert_close(
            out, expected_value * torch.ones(numel, dtype=dtype, device=self.device)
        )

    @requires_triton()
    @parametrize("nbi", [False, True])
    def test_triton_get(self, nbi: bool) -> None:
        # Configuration
        # Create symmetric tensors
        # Rank 1 gets data from rank 0 using tensor-aware API
        self._run_triton_get(nbi=nbi)

    @requires_triton()
    def test_triton_get_ring(self) -> None:
        # Configuration
        # Each rank fills its input buffer with its own rank value
        # Ring topology: each rank gets data from the rank to its left
        # rank 0 gets from rank (world_size-1), rank 1 gets from rank 0, etc.
        # All ranks execute the get operation using tensor-aware API
        self._run_triton_get_ring()

    @requires_triton()
    def test_triton_wait_until(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0
        FLAG_INITIAL_VALUE = 0
        FLAG_FINAL_VALUE = 42

        # Use a single int64 symmetric tensor as our synchronization flag.
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(
            FLAG_INITIAL_VALUE
        )
        symm_mem.rendezvous(flag, group=group_name)
        expected_flag = torch.tensor(
            [FLAG_FINAL_VALUE], dtype=torch.int32, device=self.device
        )

        if rank == 0:
            # Rank 0 (the waiter)
            my_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=FLAG_FINAL_VALUE,
            )
            # Verification
            torch.testing.assert_close(flag, expected_flag)

        if rank == 1:
            # Rank 1 (the signaler)
            # Launch a kernel to put the value to Rank 0's flag tensor.
            my_put_kernel[(1,)](flag, expected_flag, 1, peer)

    @requires_triton()
    def test_triton_signal_wait_until(self) -> None:
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        # NVSHMEM constants from documentation
        NVSHMEM_CMP_EQ = 0
        NVSHMEM_SIGNAL_SET = 0
        # Message configuration
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize
        val_to_put = 123
        completion_flag_val = 1
        flag_dtype = torch.int64

        # Producer (rank 0) prepares the data to send
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val_to_put)
        symm_mem.rendezvous(inp, group=group_name)
        # Consumer (rank 1) prepares the destination buffer
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        # Use the signal pad for synchronization, as in previous tests
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=flag_dtype).fill_(0)

        if rank == 0:
            # Producer (rank 0): Puts data into rank 1's `out` buffer and then sets the flag
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=completion_flag_val,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
            )
        elif rank == 1:
            # Consumer (rank 1): Waits on the signal variable using `signal_wait_until`.
            my_signal_wait_until_kernel[(1, 1, 1)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=completion_flag_val,
            )
            # After the wait returns, verify data and flag
            torch.testing.assert_close(
                out, val_to_put * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag,
                torch.tensor(
                    [completion_flag_val], dtype=flag_dtype, device=self.device
                ),
            )

    @requires_triton()
    def test_triton_fence(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        # Message configuration
        dtype = torch.int8
        numel = 8
        val1 = 10
        val2 = 20
        flag_val = 1
        NVSHMEM_CMP_EQ = 0

        # Symmetric buffers
        inp1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val1)
        inp2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val2)
        out1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp1, group=group_name)
        symm_mem.rendezvous(inp2, group=group_name)
        symm_mem.rendezvous(out1, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)
        # Use regular symmetric memory tensor for flag
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        symm_mem.rendezvous(flag, group=group_name)
        flag_update_val = torch.tensor(
            [flag_val], dtype=torch.int32, device=self.device
        )

        if rank == 0:
            my_put_with_fence_kernel[(1,)](
                out1,
                inp1,
                out2,
                inp2,
                flag,
                flag_update_val,
                nelems=numel,
                peer=peer,
            )
        elif rank == 1:
            # Wait until flag is set by Rank 0
            my_wait_until_kernel[(1,)](flag, cmp_op=NVSHMEM_CMP_EQ, cmp_val=flag_val)
            # Verify ordered data arrival.
            torch.testing.assert_close(
                out1, val1 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                out2, val2 * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([flag_val], dtype=torch.int32, device=self.device)
            )

    @requires_triton()
    def test_triton_quiet(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        dtype = torch.int8
        numel = 8
        val = 15
        flag_val = 42
        NVSHMEM_CMP_EQ = 0

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        flag_update_val = torch.tensor(
            [flag_val], dtype=torch.int32, device=self.device
        )

        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.rendezvous(flag, group=group_name)

        dist.barrier()
        if rank == 1:
            my_put_with_quiet_kernel[(1,)](
                out,
                inp,
                flag,
                flag_update_val,
                nelems=numel,
                peer=peer,
            )
        elif rank == 0:
            my_wait_until_kernel[(1,)](flag, cmp_op=NVSHMEM_CMP_EQ, cmp_val=flag_val)
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
        dist.barrier()

    @requires_triton()
    def test_triton_barrier(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        numel = 1
        dtype = torch.int32

        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        dst = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)

        my_barrier_test_kernel[(1,)](
            dst,
            src,
            nelems=numel,
            launch_cooperative_grid=True,
            num_ctas=1,
        )
        dist.barrier()

        if rank == 0:
            torch.testing.assert_close(
                src, torch.tensor([42], device=self.device, dtype=dtype)
            )
        else:
            torch.testing.assert_close(
                dst, torch.tensor([43], device=self.device, dtype=dtype)
            )

    @requires_triton()
    def test_triton_sync(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        numel = 1
        dtype = torch.int32

        # Create symmetric buffers
        local_data = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        remote_data = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(0)
        symm_mem.rendezvous(local_data, group=group_name)
        symm_mem.rendezvous(remote_data, group=group_name)

        # Launch kernel with cooperative grid
        my_sync_test_kernel[(1,)](
            local_data,
            remote_data,
            nelems=numel,
            launch_cooperative_grid=True,
            num_ctas=1,
        )

        # Verify results
        # Each PE should have written rank + 100 to its local_data
        expected_local = rank + 100
        torch.testing.assert_close(
            local_data, torch.tensor([expected_local], device=self.device, dtype=dtype)
        )

        next_rank = (rank + 1) % self.world_size
        # Each PE should have read (next_rank + 100) into its remote_data
        # PE 0 reads from PE 1, PE 1 reads from PE 2, ..., PE n-1 reads from PE 0
        expected_remote = next_rank + 100
        torch.testing.assert_close(
            remote_data, torch.tensor([expected_remote], device=self.device, dtype=dtype)
        )

    @requires_triton()
    def test_triton_put_signal_set(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        # Data buffers
        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        # Use the signal pad attached to the output symmetric memory handle
        # as the flag buffer for signaling completion.
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_SET = 0
        SIGNAL_VAL = 1
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
            # Rank 0 puts into Rank 1
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_SET,
                peer=peer,
            )

        if rank == 1:
            # Wait until signal flag is set by Rank 0
            my_signal_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
            )
            # After wait completes, verify data and flag contents
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([SIGNAL_VAL], dtype=torch.int64, device=self.device)
            )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        ROCM_PUT_SIGNAL_ADD_HANG_REASON,
    )
    def test_triton_put_signal_add(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        # Data buffers
        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        # Use the signal pad attached to the output symmetric memory handle
        # as the flag buffer for signaling completion.
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_ADD = 5
        SIGNAL_VAL = 16
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
            # Rank 0 puts into Rank 1
            my_putmem_signal_block_kernel[(1, 1, 1)](
                out,
                inp,
                size_bytes=msg_size_bytes,
                signal=flag,
                sig_val=SIGNAL_VAL,
                sig_op=NVSHMEM_SIGNAL_ADD,
                peer=peer,
            )

        if rank == 1:
            # Wait until signal flag is set by Rank 0
            my_signal_wait_until_kernel[(1, 1, 1)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=SIGNAL_VAL,
            )
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag, torch.tensor([SIGNAL_VAL], dtype=torch.int64, device=self.device)
            )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        ROCM_COLLECTIVE_UNAVAILABLE_REASON,
    )
    def test_triton_alltoall(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        rank = self.rank
        # Each PE will send 2 int64 elements to every other PE
        nelems_per_pe = 2
        dtype = torch.int64
        src_size = nelems_per_pe * world_size
        # Source buffer: contains data for all PEs
        # Layout: [data_for_pe0, data_for_pe1, ...]
        src = symm_mem.empty(src_size, dtype=dtype, device=self.device)
        for i in range(world_size):
            # Fill source with rank-specific data
            # Formula: rank * 100 + destination_pe
            value = rank * 100 + i
            src[i * nelems_per_pe : (i + 1) * nelems_per_pe] = value
        # Destination buffer
        dst = symm_mem.empty(src_size, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Synchronize before alltoall
        dist.barrier()
        team_handle = 0
        # Launch the kernel using new tensor-aware API
        my_alltoall_kernel[(1,)](
            team_handle,
            dst,
            src,
            nelems_per_pe,
            launch_cooperative_grid=True,
        )
        # Synchronize after alltoall
        dist.barrier()
        # Verify results
        for i in range(world_size):
            # After alltoall, we should receive data from PE i that was intended for us
            # PE i sends (i * 100 + rank) to us
            expected = i * 100 + rank
            actual = dst[i * nelems_per_pe : (i + 1) * nelems_per_pe]
            torch.testing.assert_close(actual, torch.full_like(actual, expected))

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        ROCM_COLLECTIVE_UNAVAILABLE_REASON,
    )
    def test_triton_broadcast(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        # Configuration
        nelems = 4
        dtype = torch.int64
        # Source buffer - only root will have meaningful data
        pe_root = 0
        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        # Destination buffer
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)
        if rank == pe_root:
            # Root fills with specific pattern
            for i in range(nelems):
                src[i] = 100 + i
        else:
            # Non-root PEs have dummy data
            src.fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Synchronize before broadcast
        dist.barrier()
        team_handle = 0
        # Execute broadcast
        my_broadcast_kernel[(1,)](
            team_handle,
            dst,
            src,
            nelems,
            pe_root,
            launch_cooperative_grid=True,
        )
        # Synchronize after broadcast
        dist.barrier()
        # Verify results - all ranks should have the root's data
        expected = [100 + i for i in range(nelems)]
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        ROCM_COLLECTIVE_UNAVAILABLE_REASON,
    )
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.float16,
            torch.float32,
            # torch.float64,  # Tensor-likes are not close
            torch.bfloat16,
        ],
    )
    def test_triton_sum_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        # Configuration
        nreduce = 3
        # Source buffer - each rank contributes different values
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            src[i] = (self.rank + 1) * (i + 1)
        # Destination buffer
        dst = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Calculate expected results
        expected = []
        for i in range(nreduce):
            # Sum across all ranks: sum((rank+1)*(i+1) for rank in range(world_size))
            total = sum((r + 1) * (i + 1) for r in range(world_size))
            expected.append(total)
        # Synchronize before reduction
        dist.barrier()
        team_handle = 0
        # Execute sum reduction across all ranks
        my_reduce_kernel[(1,)](
            team_handle,
            dst,
            src,
            nreduce,
            operation="sum",
            launch_cooperative_grid=True,
        )
        # Synchronize after reduction
        dist.barrier()
        # Verify results
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        ROCM_COLLECTIVE_UNAVAILABLE_REASON,
    )
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ],
    )
    def test_triton_minmax_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        # Configuration
        nreduce = 2
        # Source buffers for min and max
        src_min = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        src_max = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        # Each rank contributes different values
        # For min: rank 0: [10, 20], rank 1: [15, 5], etc.
        # For max: same values
        for i in range(nreduce):
            if i == 0:
                src_min[i] = 10 + self.rank * 5
                src_max[i] = 10 + self.rank * 5
            else:
                src_min[i] = 20 - self.rank * 15
                src_max[i] = 20 - self.rank * 15
        # Destination buffers
        dst_min = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        dst_max = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src_min, group=group_name)
        symm_mem.rendezvous(src_max, group=group_name)
        symm_mem.rendezvous(dst_min, group=group_name)
        symm_mem.rendezvous(dst_max, group=group_name)
        # Calculate expected results
        all_values = []
        for i in range(nreduce):
            values = []
            for r in range(world_size):
                if i == 0:
                    values.append(10 + r * 5)
                else:
                    values.append(20 - r * 15)
            all_values.append(values)
        expected_min = [min(vals) for vals in all_values]
        expected_max = [max(vals) for vals in all_values]
        dist.barrier()
        # Execute MIN reduction
        team_handle = 0
        # Execute MAX reduction
        my_reduce_kernel[(1,)](
            team_handle,
            dst_min,
            src_min,
            nreduce,
            operation="min",
            launch_cooperative_grid=True,
        )
        my_reduce_kernel[(1,)](
            team_handle,
            dst_max,
            src_max,
            nreduce,
            operation="max",
            launch_cooperative_grid=True,
        )
        dist.barrier()
        # Verify results
        torch.testing.assert_close(
            dst_min, torch.tensor(expected_min, device=self.device, dtype=dtype)
        )
        torch.testing.assert_close(
            dst_max, torch.tensor(expected_max, device=self.device, dtype=dtype)
        )

    @requires_triton()
    @unittest.skipIf(
        torch.version.hip is not None,
        ROCM_COLLECTIVE_UNAVAILABLE_REASON,
    )
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ],
    )
    def test_triton_prod_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        # Configuration
        nreduce = 3
        # Source buffer - each rank contributes different values
        # Use very small values to avoid overflow, especially for small integer types
        # Use values that won't overflow even for int8: all values 1 or 2
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            if i == 0:
                # For first element: rank 0,2,4... gets 1, rank 1,3,5... gets 2
                src[i] = 1 if self.rank % 2 == 0 else 2
            elif i == 1:
                # For second element: all get 1 (no multiplication effect)
                src[i] = 1
            else:
                # For third element: rank 0,1 get 1, rank 2,3 get 2, etc. (groups of 2)
                src[i] = 1 if (self.rank // 2) % 2 == 0 else 2
        # Destination buffer
        dst = symm_mem.empty(nreduce, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Calculate expected results
        vals = torch.empty(nreduce, world_size, dtype=dtype)
        vals[0, ::2] = 1
        vals[0, 1::2] = 2
        vals[1] = 1
        for rank in range(world_size):
            vals[2, rank] = 1 if (rank // 2) % 2 == 0 else 2
        expected = vals.prod(-1).tolist()
        # Synchronize before reduction
        dist.barrier()
        team_handle = 0
        # Execute product reduction across all ranks
        my_reduce_kernel[(1,)](
            team_handle,
            dst,
            src,
            nreduce,
            operation="prod",
            launch_cooperative_grid=True,
        )
        # Synchronize after reduction
        dist.barrier()
        # Verify results
        torch.testing.assert_close(
            dst, torch.tensor(expected, device=self.device, dtype=dtype)
        )

    @requires_triton()
    def test_triton_fused_get_add(self) -> None:
        self._init_device()
        world_size = dist.get_world_size()
        if world_size != 2:
            self.skipTest("Fused get+add test currently expects world_size=2")

        group_name = dist.distributed_c10d._get_default_group().group_name
        nelems = 256
        dtype = torch.float32
        peer = 1 - self.rank

        local_inp = torch.arange(nelems, device=self.device, dtype=dtype) + (
            1000 * self.rank
        )
        remote_symm_src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        remote_symm_src.copy_(local_inp)
        symm_mem.rendezvous(remote_symm_src, group=group_name)

        tmp_remote = torch.empty(nelems, device=self.device, dtype=dtype)
        out = torch.empty(nelems, device=self.device, dtype=dtype)

        my_fused_get_add_kernel[(1,)](
            out,
            local_inp,
            remote_symm_src,
            tmp_remote,
            nelems,
            peer,
            block_size=256,
        )

        expected = local_inp + (
            torch.arange(nelems, device=self.device, dtype=dtype) + (1000 * peer)
        )
        torch.testing.assert_close(out, expected)

    @requires_triton()
    def test_triton_fused_get_add_benchmark_vs_rccl(self) -> None:
        if os.getenv("SHMEM_TRITON_BENCHMARK", "0") != "1":
            return

        self._init_device()
        world_size = dist.get_world_size()
        if world_size != 2:
            self.skipTest("Benchmark currently expects world_size=2")

        group_name = dist.distributed_c10d._get_default_group().group_name
        dtype = torch.float32
        # Current fused kernel is single-program; keep problem size <= block_size.
        nelems = 1024
        warmup = 5
        iters = 20
        peer = 1 - self.rank

        local_inp = torch.randn(nelems, device=self.device, dtype=dtype)
        remote_symm_src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        remote_symm_src.copy_(local_inp)
        symm_mem.rendezvous(remote_symm_src, group=group_name)

        tmp_remote = torch.empty(nelems, device=self.device, dtype=dtype)
        out = torch.empty(nelems, device=self.device, dtype=dtype)

        # RCCL apples-to-apples baseline:
        # all_gather (communication) + explicit local add (compute).
        gather_buf = torch.empty(world_size * nelems, device=self.device, dtype=dtype)
        rccl_out = torch.empty_like(local_inp)
        peer_chunk = gather_buf[peer * nelems : (peer + 1) * nelems]

        def _run_rccl_allgather_add() -> None:
            dist.all_gather_into_tensor(gather_buf, local_inp)
            torch.add(local_inp, peer_chunk, out=rccl_out)

        # Correctness check against RCCL all_gather + add for world_size=2.
        my_fused_get_add_kernel[(1,)](
            out,
            local_inp,
            remote_symm_src,
            tmp_remote,
            nelems,
            peer,
            block_size=1024,
        )
        _run_rccl_allgather_add()
        torch.testing.assert_close(out, rccl_out, atol=1e-5, rtol=1e-5)

        def _bench_shmem_ms() -> float:
            dist.barrier()
            start = device_module.Event(enable_timing=True)
            end = device_module.Event(enable_timing=True)
            for _ in range(warmup):
                my_fused_get_add_kernel[(1,)](
                    out,
                    local_inp,
                    remote_symm_src,
                    tmp_remote,
                    nelems,
                    peer,
                    block_size=1024,
                )
            end.record()
            end.synchronize()
            dist.barrier()

            start.record()
            for _ in range(iters):
                my_fused_get_add_kernel[(1,)](
                    out,
                    local_inp,
                    remote_symm_src,
                    tmp_remote,
                    nelems,
                    peer,
                    block_size=1024,
                )
            end.record()
            end.synchronize()
            dist.barrier()
            return float(start.elapsed_time(end) / iters)

        def _bench_rccl_allgather_add_ms() -> float:
            dist.barrier()
            start = device_module.Event(enable_timing=True)
            end = device_module.Event(enable_timing=True)
            for _ in range(warmup):
                _run_rccl_allgather_add()
            end.record()
            end.synchronize()
            dist.barrier()

            start.record()
            for _ in range(iters):
                _run_rccl_allgather_add()
            end.record()
            end.synchronize()
            dist.barrier()
            return float(start.elapsed_time(end) / iters)

        shmem_ms = _bench_shmem_ms()
        rccl_ms = _bench_rccl_allgather_add_ms()

        shmem_ms_t = torch.tensor([shmem_ms], device=self.device, dtype=torch.float32)
        rccl_ms_t = torch.tensor([rccl_ms], device=self.device, dtype=torch.float32)
        dist.all_reduce(shmem_ms_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(rccl_ms_t, op=dist.ReduceOp.SUM)
        shmem_ms_mean = (shmem_ms_t / world_size).item()
        rccl_ms_mean = (rccl_ms_t / world_size).item()

        if self.rank == 0:
            speedup = rccl_ms_mean / shmem_ms_mean if shmem_ms_mean > 0 else float("inf")
            print(
                f"[fused_get_add_bench] nelems={nelems} dtype={dtype} "
                f"shmem_fused_get_add_ms={shmem_ms_mean:.3f} "
                f"rccl_allgather_add_ms={rccl_ms_mean:.3f} "
                f"speedup_vs_rccl_allgather_add={speedup:.3f}x"
            )

    @requires_triton()
    def test_triton_matmul_all_gather_benchmark_vs_rccl(self) -> None:
        # Opt-in benchmark to keep default test runtime unchanged.
        if os.getenv("SHMEM_TRITON_MATMUL_AG_BENCHMARK", "0") != "1":
            return

        self._init_device()
        world_size = dist.get_world_size()
        rank = self.rank

        # Match Iris smoke defaults unless overridden.
        m_local = int(os.getenv("SHMEM_TRITON_BENCH_M_LOCAL", "1024"))
        n = int(os.getenv("SHMEM_TRITON_BENCH_N", "3584"))
        k = int(os.getenv("SHMEM_TRITON_BENCH_K", "8192"))
        warmup = int(os.getenv("SHMEM_TRITON_BENCH_WARMUP", "5"))
        iters = int(os.getenv("SHMEM_TRITON_BENCH_ITERS", "20"))
        dtype = torch.float16

        group_name = dist.distributed_c10d._get_default_group().group_name
        m_global = m_local * world_size
        chunk_elems = m_local * n
        chunk_start = rank * chunk_elems
        chunk_end = chunk_start + chunk_elems

        # Deterministic data; Iris smoke uses A filled with 1.0.
        torch.manual_seed(0)
        a_local = torch.ones((m_local, k), device=self.device, dtype=dtype)
        b = torch.randn((k, n), device=self.device, dtype=dtype)

        c_gather_shmem = symm_mem.empty((m_global, n), device=self.device, dtype=dtype)
        symm_mem.rendezvous(c_gather_shmem, group=group_name)
        c_gather_shmem_flat = c_gather_shmem.view(-1)
        # Compute directly into the local shard in symmetric output to avoid extra copy.
        c_local_shmem = c_gather_shmem[rank * m_local : (rank + 1) * m_local]
        c_local_rccl = torch.empty((m_local, n), device=self.device, dtype=dtype)

        c_gather_rccl = torch.empty((m_global, n), device=self.device, dtype=dtype)

        local_chunk_shmem = c_gather_shmem_flat[chunk_start:chunk_end]

        shmem_mode = os.getenv("SHMEM_TRITON_MATMUL_AG_MODE", "two_phase_put")
        if shmem_mode == "fused_tritonblas":
            default_block_m = "32"
            default_block_n = "1024"
            default_block_k = "16"
            default_num_warps = "4"
            default_group_size_m = "4"
        elif shmem_mode in ("tritonblas_two_phase", "fused_tritonblas_bulk"):
            default_block_m = "128"
            default_block_n = "128"
            default_block_k = "32"
            default_num_warps = "4"
            default_group_size_m = "4"
        else:
            default_block_m = "8"
            default_block_n = "128"
            default_block_k = "32"
            default_num_warps = "4"
            default_group_size_m = "8"
        block_m = int(os.getenv("SHMEM_TRITON_BENCH_BLOCK_M", default_block_m))
        block_n = int(os.getenv("SHMEM_TRITON_BENCH_BLOCK_N", default_block_n))
        block_k = int(os.getenv("SHMEM_TRITON_BENCH_BLOCK_K", default_block_k))
        num_warps = int(os.getenv("SHMEM_TRITON_BENCH_NUM_WARPS", default_num_warps))
        group_size_m = int(os.getenv("SHMEM_TRITON_BENCH_GROUP_SIZE_M", default_group_size_m))
        num_pid_m = triton.cdiv(m_local, block_m)
        num_pid_n = triton.cdiv(n, block_n)
        total_tiles = num_pid_m * num_pid_n
        num_programs = int(os.getenv("SHMEM_TRITON_BENCH_NUM_PROGRAMS", str(min(120, total_tiles))))
        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        even_k = k % block_k == 0
        even_n = n % block_n == 0
        peer = 1 - rank if world_size == 2 else -1
        num_put_wgs = int(os.getenv("SHMEM_TRITON_BENCH_NUM_PUT_WGS", "64"))
        put_chunk_per_wg = triton.cdiv(chunk_elems, num_put_wgs)
        multi_peer_mode = os.getenv("SHMEM_TRITON_BENCH_MULTI_PEER_MODE", "striped")
        if world_size > 2 and multi_peer_mode == "striped":
            num_peers = world_size - 1
            put_chunk_per_wg_per_peer = triton.cdiv(chunk_elems, num_put_wgs)
            striped_grid_size = num_put_wgs * num_peers
        else:
            num_peers = max(world_size - 1, 1)
            put_chunk_per_wg_per_peer = put_chunk_per_wg
            striped_grid_size = num_put_wgs

        done_counter = torch.zeros(1, dtype=torch.int32, device=self.device)

        def _run_shmem_matmul_all_gather():
            if shmem_mode == "fused_naive":
                if world_size != 2:
                    raise RuntimeError("fused_naive mode only supports world_size=2.")
                grid = (triton.cdiv(m_local, block_m), triton.cdiv(n, block_n))
                my_fused_matmul_allgather_put_kernel[grid](
                    a_local,
                    b,
                    c_gather_shmem,
                    m_local,
                    n,
                    k,
                    a_local.stride(0),
                    a_local.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c_gather_shmem.stride(0),
                    c_gather_shmem.stride(1),
                    rank,
                    peer,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=num_warps,
                )
                # One quiet per iteration to ensure visibility before next iteration.
                my_quiet_kernel[(1,)]()
            elif shmem_mode == "fused_persistent":
                if world_size != 2:
                    raise RuntimeError("fused_persistent mode only supports world_size=2.")
                grid = (num_programs,)
                my_fused_matmul_allgather_put_persistent_kernel[grid](
                    a_local,
                    b,
                    c_gather_shmem,
                    m_local,
                    n,
                    k,
                    a_local.stride(0),
                    a_local.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c_gather_shmem.stride(0),
                    c_gather_shmem.stride(1),
                    rank,
                    peer,
                    num_pid_n,
                    total_tiles,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=num_warps,
                )
                my_quiet_kernel[(1,)]()
            elif shmem_mode == "fused_tritonblas":
                if world_size != 2:
                    raise RuntimeError("fused_tritonblas mode only supports world_size=2.")
                if not HAS_TRITONBLAS:
                    raise RuntimeError(
                        "fused_tritonblas mode requested but tritonblas is unavailable in this venv."
                    )
                grid = (num_sms,)
                my_fused_matmul_allgather_put_tritonblas_kernel[grid](
                    a_local,
                    b,
                    c_gather_shmem,
                    m_local,
                    n,
                    k,
                    a_local.stride(0),
                    a_local.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c_gather_shmem.stride(0),
                    c_gather_shmem.stride(1),
                    rank,
                    peer,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_sms=num_sms,
                    group_size_m=group_size_m,
                    even_k=even_k,
                    even_n=even_n,
                    num_warps=num_warps,
                )
                my_quiet_kernel[(1,)]()
            elif shmem_mode == "tritonblas_two_phase":
                if not HAS_TRITONBLAS:
                    raise RuntimeError(
                        "tritonblas_two_phase mode requested but tritonblas is unavailable."
                    )
                grid = (num_sms,)
                my_tritonblas_gemm_only_kernel[grid](
                    a_local,
                    b,
                    c_gather_shmem,
                    m_local,
                    n,
                    k,
                    a_local.stride(0),
                    a_local.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c_gather_shmem.stride(0),
                    c_gather_shmem.stride(1),
                    rank,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_sms=num_sms,
                    group_size_m=group_size_m,
                    even_k=even_k,
                    num_warps=num_warps,
                )
                if world_size == 2:
                    my_parallel_put_quiet_kernel[(num_put_wgs,)](
                        local_chunk_shmem, chunk_elems, peer,
                        chunk_per_wg=put_chunk_per_wg,
                    )
                else:
                    if multi_peer_mode == "striped":
                        my_parallel_put_striped_peers_quiet_kernel[(striped_grid_size,)](
                            local_chunk_shmem, chunk_elems, rank,
                            chunk_per_wg_per_peer=put_chunk_per_wg_per_peer,
                            world_size=world_size,
                            wgs_per_peer=num_put_wgs,
                        )
                    else:
                        my_parallel_put_allpeers_quiet_kernel[(num_put_wgs,)](
                            local_chunk_shmem, chunk_elems, rank,
                            chunk_per_wg=put_chunk_per_wg,
                            world_size=world_size,
                        )
            elif shmem_mode == "fused_tritonblas_bulk":
                if world_size != 2:
                    raise RuntimeError(
                        "fused_tritonblas_bulk mode only supports world_size=2 (single peer put)."
                    )
                if not HAS_TRITONBLAS:
                    raise RuntimeError(
                        "fused_tritonblas_bulk mode requested but tritonblas is unavailable."
                    )
                done_counter.zero_()
                grid = (num_sms,)
                my_fused_matmul_allgather_bulk_tritonblas_kernel[grid](
                    a_local,
                    b,
                    c_gather_shmem,
                    m_local,
                    n,
                    k,
                    a_local.stride(0),
                    a_local.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c_gather_shmem.stride(0),
                    c_gather_shmem.stride(1),
                    rank,
                    peer,
                    chunk_elems,
                    done_counter,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_sms=num_sms,
                    group_size_m=group_size_m,
                    even_k=even_k,
                    num_warps=num_warps,
                )
            elif shmem_mode == "origami_two_phase":
                if not HAS_TRITONBLAS_MATMUL:
                    raise RuntimeError(
                        "origami_two_phase mode requires tritonblas package with matmul()."
                    )
                _tritonblas_mod.matmul(a_local, b, c_local_shmem)
                if world_size == 2:
                    my_parallel_put_quiet_kernel[(num_put_wgs,)](
                        local_chunk_shmem, chunk_elems, peer,
                        chunk_per_wg=put_chunk_per_wg,
                    )
                elif multi_peer_mode == "striped":
                    my_parallel_put_striped_peers_quiet_kernel[(striped_grid_size,)](
                        local_chunk_shmem, chunk_elems, rank,
                        chunk_per_wg_per_peer=put_chunk_per_wg_per_peer,
                        world_size=world_size,
                        wgs_per_peer=num_put_wgs,
                    )
                else:
                    my_parallel_put_allpeers_quiet_kernel[(num_put_wgs,)](
                        local_chunk_shmem, chunk_elems, rank,
                        chunk_per_wg=put_chunk_per_wg,
                        world_size=world_size,
                    )
            else:
                # two_phase_put: rocBLAS GEMM + parallel multi-WG SHMEM put.
                torch.mm(a_local, b, out=c_local_shmem)
                if world_size == 2:
                    my_parallel_put_quiet_kernel[(num_put_wgs,)](
                        local_chunk_shmem, chunk_elems, peer,
                        chunk_per_wg=put_chunk_per_wg,
                    )
                elif multi_peer_mode == "striped":
                    my_parallel_put_striped_peers_quiet_kernel[(striped_grid_size,)](
                        local_chunk_shmem, chunk_elems, rank,
                        chunk_per_wg_per_peer=put_chunk_per_wg_per_peer,
                        world_size=world_size,
                        wgs_per_peer=num_put_wgs,
                    )
                elif multi_peer_mode == "per_peer_seq":
                    for p in range(world_size):
                        if p != rank:
                            my_parallel_put_no_quiet_kernel[(num_put_wgs,)](
                                local_chunk_shmem, chunk_elems, p,
                                chunk_per_wg=put_chunk_per_wg,
                            )
                    my_quiet_kernel[(1,)]()
                else:
                    my_parallel_put_allpeers_quiet_kernel[(num_put_wgs,)](
                        local_chunk_shmem, chunk_elems, rank,
                        chunk_per_wg=put_chunk_per_wg,
                        world_size=world_size,
                    )

        def _run_rccl_matmul_all_gather():
            torch.mm(a_local, b, out=c_local_rccl)
            dist.all_gather_into_tensor(c_gather_rccl, c_local_rccl)

        # Correctness check (single iteration).
        _run_shmem_matmul_all_gather()
        dist.barrier()
        _run_rccl_matmul_all_gather()
        torch.testing.assert_close(c_gather_shmem, c_gather_rccl, atol=1e-2, rtol=1e-2)

        def _bench_ms(run_fn) -> float:
            dist.barrier()
            for _ in range(warmup):
                run_fn()
            dist.barrier()

            start = device_module.Event(enable_timing=True)
            end = device_module.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                run_fn()
            end.record()
            end.synchronize()
            dist.barrier()
            return float(start.elapsed_time(end) / iters)

        shmem_ms = _bench_ms(_run_shmem_matmul_all_gather)
        rccl_ms = _bench_ms(_run_rccl_matmul_all_gather)

        # Average per-rank timing.
        shmem_ms_t = torch.tensor([shmem_ms], device=self.device, dtype=torch.float32)
        rccl_ms_t = torch.tensor([rccl_ms], device=self.device, dtype=torch.float32)
        dist.all_reduce(shmem_ms_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(rccl_ms_t, op=dist.ReduceOp.SUM)
        shmem_ms_mean = (shmem_ms_t / world_size).item()
        rccl_ms_mean = (rccl_ms_t / world_size).item()

        # Match Iris accounting.
        flops = 2 * m_local * n * k
        nbytes = (world_size - 1) * m_local * n * torch.finfo(dtype).bits // 8

        def _metrics(ms: float):
            sec = ms * 1.0e-3
            tflops = (flops / 1.0e12) / sec if sec > 0 else float("inf")
            bw_gbps = (nbytes / 1.0e9) / sec if sec > 0 else float("inf")
            return bw_gbps, tflops

        shmem_bw, shmem_tflops = _metrics(shmem_ms_mean)
        rccl_bw, rccl_tflops = _metrics(rccl_ms_mean)

        if rank == 0:
            speedup = rccl_ms_mean / shmem_ms_mean if shmem_ms_mean > 0 else float("inf")
            parity = shmem_ms_mean / rccl_ms_mean if rccl_ms_mean > 0 else float("inf")
            print(
                f"\n[matmul_all_gather_bench] world_size={world_size} "
                f"M_local={m_local} N={n} K={k} dtype={dtype}\n"
                f"  mode={shmem_mode} "
                f"tiles=({block_m}x{block_n}x{block_k}) "
                f"warps={num_warps} group_m={group_size_m} "
                f"put_wgs={num_put_wgs} multi_peer={multi_peer_mode}\n"
                f"  shmem: {shmem_ms_mean:.4f} ms | "
                f"{shmem_tflops:.2f} TFLOPS | {shmem_bw:.2f} GB/s\n"
                f"  rccl:  {rccl_ms_mean:.4f} ms | "
                f"{rccl_tflops:.2f} TFLOPS | {rccl_bw:.2f} GB/s\n"
                f"  parity={parity:.3f}x (1.0=matched) "
                f"speedup_vs_rccl={speedup:.3f}x"
            )

            out_csv = os.getenv(
                "SHMEM_TRITON_MATMUL_AG_CSV",
                "benchmark_matmul_all_gather_torch_triton_rocshmem.csv",
            )
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "benchmark",
                        "world_size",
                        "num_ranks",
                        "M_local",
                        "N",
                        "K",
                        "dtype",
                        "gpu_time_ms",
                        "bandwidth_gbps",
                        "tflops",
                        "skipped",
                        "skip_reason",
                    ]
                )
                writer.writerow(
                    [
                        "torch_triton_rocshmem_matmul_all_gather",
                        world_size,
                        world_size,
                        m_local,
                        n,
                        k,
                        "float16",
                        f"{shmem_ms_mean:.4f}",
                        f"{shmem_bw:.2f}",
                        f"{shmem_tflops:.2f}",
                        "False",
                        "",
                    ]
                )
                writer.writerow(
                    [
                        "torch_rccl_matmul_all_gather",
                        world_size,
                        world_size,
                        m_local,
                        n,
                        k,
                        "float16",
                        f"{rccl_ms_mean:.4f}",
                        f"{rccl_bw:.2f}",
                        f"{rccl_tflops:.2f}",
                        "False",
                        "",
                    ]
                )


instantiate_parametrized_tests(ShmemTritonTestBase)


class SHMEMTritonTest(ShmemTritonTestBase):
    __test__ = True

SHMEMTritonTest = skip_but_pass_in_sandcastle_if(
    torch.version.hip is None and not IS_H100,
    "NVSHMEM Triton tests require H100.",
)(SHMEMTritonTest)


if __name__ == "__main__":
    run_tests()
