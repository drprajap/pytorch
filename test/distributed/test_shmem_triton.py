# Owner(s): ["oncall: distributed"]

import unittest

import triton.language as tl

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._inductor.runtime.triton_compat import triton
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.inductor_utils import IS_H100, requires_triton
from torch.utils._triton import has_triton


def _requires_unavailable_backend(jit_func):  # type: ignore[no-untyped-def]
    return jit_func


if torch.version.hip is not None:
    import torch.distributed._symmetric_memory._rocshmem_triton as shmem_backend
    from torch.distributed._symmetric_memory._rocshmem_triton import (
        requires_rocshmem as requires_shmem,
    )
else:
    import torch.distributed._symmetric_memory._nvshmem_triton as shmem_backend
    from torch.distributed._symmetric_memory._nvshmem_triton import (
        requires_nvshmem as requires_shmem,
    )


device_type = "cuda"
device_module = torch.get_device_module(device_type)


class ShmemBackendMixin:
    backend_name = "NVSHMEM"

    def is_backend_available(self) -> bool:
        return symm_mem.is_nvshmem_available()

    def skip_or_xfail_reason(self, op: str) -> str | None:
        return None

    def requires_h100(self) -> bool:
        return False


class NVShmemBackendMixin(ShmemBackendMixin):
    backend_name = "NVSHMEM"

    def requires_h100(self) -> bool:
        return True

    def skip_or_xfail_reason(self, op: str) -> str | None:
        if self.requires_h100() and not IS_H100:
            return "NVSHMEM Triton tests require H100."
        return None


class RocShmemBackendMixin(ShmemBackendMixin):
    # Same backend id as CUDA NVSHMEM path; implementation is rocSHMEM on HIP.
    backend_name = "NVSHMEM"

    def skip_or_xfail_reason(self, op: str) -> str | None:
        if torch.version.hip is None:
            return "rocSHMEM Triton tests only run on ROCm."
        if op in {
            "test_triton_alltoall",
            "test_triton_broadcast",
            "test_triton_sum_reduce",
            "test_triton_minmax_reduce",
            "test_triton_prod_reduce",
        }:
            return (
                "rocSHMEM *_wg collective symbols are unavailable in current device bitcode for this op."
            )
        if op == "test_triton_put_signal_add":
            return "Known hang in rocSHMEM Triton put_signal_add path."
        return None


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
    shmem_backend.putmem_signal_block(dst, src, size_bytes, signal, sig_val, sig_op, peer)


@requires_shmem
@triton.jit
def my_signal_wait_until_kernel(signal, cmp_op, cmp_val):
    shmem_backend.signal_wait_until(signal, cmp_op, cmp_val)


@requires_shmem
@triton.jit
def my_wait_until_kernel(ivar, cmp_op, cmp_val):
    shmem_backend.wait_until(ivar, cmp_op, cmp_val)


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
    shmem_backend.put(dst1, src1, nelems, peer)
    shmem_backend.fence()
    shmem_backend.put(dst2, src2, nelems, peer)
    shmem_backend.fence()
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
    shmem_backend.put(dst, src, nelems, peer)
    shmem_backend.quiet()
    shmem_backend.put(flag_dst, flag_src, 1, peer)


class ShmemTritonTestBase(MultiProcContinuousTest):
    backend_name = "NVSHMEM"

    def setUp(self) -> None:
        super().setUp()
        if self.__class__ is ShmemTritonTestBase:
            self.skipTest("Abstract SHMEM base test class")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _skip_unavailable(self, op_name: str) -> None:
        if not has_triton():
            self.skipTest("Triton not available")
        if not self.is_backend_available():
            self.skipTest(f"{self.backend_name}/rocSHMEM backend not available")
        reason = self.skip_or_xfail_reason(op_name)
        if reason:
            self.skipTest(reason)

    def _init_device(self) -> None:
        device_module.set_device(self.device)
        symm_mem.set_backend(self.backend_name)

    @requires_triton()
    def test_triton_put(self) -> None:
        self._skip_unavailable("test_triton_put")
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        nelems = 5
        dtype = torch.int64
        val = 42 + rank

        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)
        for i in range(nelems):
            src[i] = val * 10 + i

        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        dist.barrier()

        peer = 1 - rank
        if rank == 0:
            my_put_kernel[(1,)](dst, src, nelems, peer)

        dist.barrier()
        if rank == 1:
            expected = [420 + i for i in range(nelems)]
            torch.testing.assert_close(
                dst, torch.tensor(expected, device=self.device, dtype=dtype)
            )

    def _run_triton_get(self, nbi: bool) -> None:
        self._skip_unavailable("test_triton_get")
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        numel = 8
        dtype = torch.int8
        val = 7

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        peer = 1 - rank
        if rank == 1:
            my_get_kernel[(1,)](out, inp, numel, peer, nbi=nbi)

        if rank == 1:
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

    def _run_triton_get_ring(self) -> None:
        self._skip_unavailable("test_triton_get_ring")
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        world_size = dist.get_world_size()
        numel = 8
        dtype = torch.int8

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(rank)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        peer = (rank - 1) % world_size
        my_get_kernel[(1,)](out, inp, numel, peer, nbi=False)

        expected_value = peer
        torch.testing.assert_close(
            out, expected_value * torch.ones(numel, dtype=dtype, device=self.device)
        )

    @requires_triton()
    def test_triton_get(self) -> None:
        self._run_triton_get(nbi=False)

    @requires_triton()
    def test_triton_get_nbi(self) -> None:
        self._run_triton_get(nbi=True)

    @requires_triton()
    def test_triton_get_ring(self) -> None:
        self._run_triton_get_ring()

    @requires_triton()
    def test_triton_wait_until(self) -> None:
        self._skip_unavailable("test_triton_wait_until")
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0
        FLAG_INITIAL_VALUE = 0
        FLAG_FINAL_VALUE = 42

        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(
            FLAG_INITIAL_VALUE
        )
        symm_mem.rendezvous(flag, group=group_name)
        expected_flag = torch.tensor(
            [FLAG_FINAL_VALUE], dtype=torch.int32, device=self.device
        )

        if rank == 0:
            my_wait_until_kernel[(1,)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=FLAG_FINAL_VALUE,
            )
            torch.testing.assert_close(flag, expected_flag)

        if rank == 1:
            my_put_kernel[(1,)](flag, expected_flag, 1, peer)

    @requires_triton()
    def test_triton_signal_wait_until(self) -> None:
        self._skip_unavailable("test_triton_signal_wait_until")
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        NVSHMEM_CMP_EQ = 0
        NVSHMEM_SIGNAL_SET = 0
        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize
        val_to_put = 123
        completion_flag_val = 1
        flag_dtype = torch.int64

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val_to_put)
        symm_mem.rendezvous(inp, group=group_name)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=flag_dtype).fill_(0)

        if rank == 0:
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
            my_signal_wait_until_kernel[(1, 1, 1)](
                flag,
                cmp_op=NVSHMEM_CMP_EQ,
                cmp_val=completion_flag_val,
            )
            torch.testing.assert_close(
                out, val_to_put * torch.ones(numel, dtype=dtype, device=self.device)
            )
            torch.testing.assert_close(
                flag,
                torch.tensor([completion_flag_val], dtype=flag_dtype, device=self.device),
            )

    @requires_triton()
    def test_triton_fence(self) -> None:
        self._skip_unavailable("test_triton_fence")
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank
        peer = 1 - rank
        dtype = torch.int8
        numel = 8
        val1 = 10
        val2 = 20
        flag_val = 1
        NVSHMEM_CMP_EQ = 0

        inp1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val1)
        inp2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val2)
        out1 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        out2 = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp1, group=group_name)
        symm_mem.rendezvous(inp2, group=group_name)
        symm_mem.rendezvous(out1, group=group_name)
        symm_mem.rendezvous(out2, group=group_name)
        flag = symm_mem.empty(1, dtype=torch.int32, device=self.device).fill_(0)
        symm_mem.rendezvous(flag, group=group_name)
        flag_update_val = torch.tensor([flag_val], dtype=torch.int32, device=self.device)

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
            my_wait_until_kernel[(1,)](flag, cmp_op=NVSHMEM_CMP_EQ, cmp_val=flag_val)
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
        self._skip_unavailable("test_triton_quiet")
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
        flag_update_val = torch.tensor([flag_val], dtype=torch.int32, device=self.device)

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
    def test_triton_put_signal_set(self) -> None:
        self._skip_unavailable("test_triton_put_signal_set")
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_SET = 0
        SIGNAL_VAL = 1
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
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
            my_signal_wait_until_kernel[(1,)](
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
    def test_triton_put_signal_add(self) -> None:
        self._skip_unavailable("test_triton_put_signal_add")
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        msg_size_bytes = 8
        dtype = torch.int8
        numel = msg_size_bytes // dtype.itemsize

        val = 11
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(val)
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        out_hdl = symm_mem.rendezvous(out, group=group_name)
        flag = out_hdl.get_signal_pad(rank, (1,), dtype=torch.int64).fill_(0)

        peer = 1 - rank
        NVSHMEM_SIGNAL_ADD = 5
        SIGNAL_VAL = 16
        NVSHMEM_CMP_EQ = 0

        if rank == 0:
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

    @unittest.skip("Collective launch stays backend-specific for now.")
    @requires_triton()
    def test_triton_alltoall(self) -> None:
        pass

    @unittest.skip("Collective launch stays backend-specific for now.")
    @requires_triton()
    def test_triton_broadcast(self) -> None:
        pass

    @unittest.skip("Collective launch stays backend-specific for now.")
    @requires_triton()
    def test_triton_sum_reduce(self) -> None:
        pass

    @unittest.skip("Collective launch stays backend-specific for now.")
    @requires_triton()
    def test_triton_minmax_reduce(self) -> None:
        pass

    @unittest.skip("Collective launch stays backend-specific for now.")
    @requires_triton()
    def test_triton_prod_reduce(self) -> None:
        pass

