# Owner(s): ["oncall: distributed"]
#
# ROCm / rocSHMEM Triton tests
# TODO: Enable Collective device tests (alltoall, broadcast, team reduce)
#       Currently these tests are skipped on ROCm because rocSHMEM *_wg
#       collective symbols are not in current device bitcode.
#       See torch/distributed/_symmetric_memory/_rocshmem_triton.py.
#
# Run (2 GPUs):
#   HIP_VISIBLE_DEVICES=0,1 python -m pytest test/distributed/test_rocshmem_triton.py -v -k "not test_collective_device"

import unittest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)
from torch.utils._triton import has_triton
from test_shmem_triton import (
    RocShmemBackendMixin,
    ShmemTritonTestBase,
    my_putmem_signal_block_kernel,
    my_signal_wait_until_kernel,
)

def requires_rocshmem_available():
    on_rocm = torch.version.hip is not None
    return skip_but_pass_in_sandcastle_if(
        not on_rocm or not symm_mem.is_nvshmem_available(),
        "test requires ROCm build with rocSHMEM",
    )


if has_triton() and torch.version.hip is not None:
    import triton.language as tl
    import torch.distributed._symmetric_memory._rocshmem_triton as rocshmem
    from torch._inductor.runtime.triton_compat import triton
    from torch.testing._internal.inductor_utils import requires_triton
    from torch.distributed._symmetric_memory._rocshmem_triton import requires_rocshmem

    # Shared Triton JIT kernels

    @requires_rocshmem
    @triton.jit
    def my_barrier_test_kernel(
        dst,
        src,
        nelems,
    ):
        # Testing barrier_all() requires coordinated operations across PEs within
        # the same kernel execution. Unlike other kernels that just wrap device SHMEM
        # primitives, this one implements the full test logic to properly verify
        # device-side barrier synchronization.
        my_pe = rocshmem.my_pe()
        n_pes = rocshmem.n_pes()
    
        # Rank 0 broadcasts its value to all other ranks
        if my_pe == 0:
            # Write initial value
            p_src = src.to(tl.pointer_type(tl.int32))
            tl.store(p_src, 42)
            # Put to all other ranks
            i = 1
            while i < n_pes:
                rocshmem.put(dst, src, nelems, i)
                i += 1
    
        # Synchronize all PEs
        rocshmem.barrier_all()
    
        # Non-zero ranks increment the received value
        if my_pe != 0:
            p_dst = dst.to(tl.pointer_type(tl.int32))
            received = tl.load(p_dst)
            tl.store(p_dst, received + 1)
    
    
    @requires_rocshmem
    @triton.jit
    def my_sync_test_kernel(
        local_data,
        remote_data,
        nelems,
    ):
        my_pe = rocshmem.my_pe()
        n_pes = rocshmem.n_pes()
    
        # Each PE writes a unique value to its local memory
        p_local = local_data.to(tl.pointer_type(tl.int32))
        unique_value = my_pe + 100  # PE 0 writes 100, PE 1 writes 101, etc.
        tl.store(p_local, unique_value)
    
        # sync_all() ensures local stores are visible to other PEs
        # but doesn't guarantee completion of any remote operations
        rocshmem.sync_all()
    
        # Now each PE reads from the next PE's memory to verify visibility
        # PE 0 reads from PE 1, PE 1 reads from PE 2, ..., PE n-1 reads from PE 0
        next_pe = (my_pe + 1) % n_pes
        rocshmem.get(remote_data, local_data, nelems, next_pe)
    
        # The get should now see the value that the next PE wrote locally
        # because sync_all() made those local stores visible
    
    
    @requires_rocshmem
    @triton.jit
    def my_alltoall_kernel(
        team_handle,
        dst,
        src,
        nelems_per_pe,
    ):
        rocshmem.alltoall(team_handle, dst, src, nelems_per_pe)
    
    
    @requires_rocshmem
    @triton.jit
    def my_broadcast_kernel(
        team_handle,
        dst,
        src,
        nelems,
        pe_root,
    ):
        rocshmem.broadcast(team_handle, dst, src, nelems, pe_root)
    
    
    @requires_rocshmem
    @triton.jit
    def my_reduce_kernel(
        team_handle,
        dest_tensor,
        source_tensor,
        nreduce,
        operation: tl.constexpr,
    ):
        rocshmem.reduce(team_handle, dest_tensor, source_tensor, nreduce, operation)
    
    

@requires_rocshmem_available()
@instantiate_parametrized_tests
class ROCSHMEMTritonTest(RocShmemBackendMixin, ShmemTritonTestBase):

    @unittest.skip("Known hang in rocSHMEM Triton put_signal_add path.")
    @requires_triton()
    def test_triton_put_signal_add(self) -> None:
        pass

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

        # Each PE should have read (next_rank + 100) into its remote_data
        # PE 0 reads from PE 1, PE 1 reads from PE 2, ..., PE n-1 reads from PE 0
        next_rank = (rank + 1) % self.world_size
        expected_remote = next_rank + 100
        torch.testing.assert_close(
            remote_data,
            torch.tensor([expected_remote], device=self.device, dtype=dtype),
        )

    @unittest.skip(
        "rocSHMEM collective symbol rocshmem_alltoallmem_wg is unavailable in current device bitcode; see _rocshmem_triton.py"
    )
    @requires_triton()
    def test_triton_alltoall(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        rank = self.rank
        # Each PE will send 2 int64 elements to every other PE
        nelems_per_pe = 2
        dtype = torch.int64
        # Source buffer: contains data for all PEs
        # Layout: [data_for_pe0, data_for_pe1, ...]
        src_size = nelems_per_pe * world_size
        src = symm_mem.empty(src_size, dtype=dtype, device=self.device)
        # Fill source with rank-specific data
        # Formula: rank * 100 + destination_pe
        for i in range(world_size):
            value = rank * 100 + i
            src[i * nelems_per_pe : (i + 1) * nelems_per_pe] = value
        # Destination buffer
        dst = symm_mem.empty(src_size, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)
        # Synchronize before alltoall
        dist.barrier()
        team_handle = 0  # NVSHMEM_TEAM_WORLD handle is 0
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

    @unittest.skip(
        "rocSHMEM collective symbol rocshmem_broadcastmem_wg is unavailable in current device bitcode; see _rocshmem_triton.py"
    )
    @requires_triton()
    def test_triton_broadcast(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        # Configuration
        nelems = 4  # number of elements
        dtype = torch.int64

        # Source buffer - only root will have meaningful data
        pe_root = 0  # PE 0 will be the root
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

        # Execute broadcast
        team_handle = 0  # NVSHMEM_TEAM_WORLD
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

    @unittest.skip(
        "rocSHMEM team-reduce collective symbols are unavailable in current device bitcode; see _rocshmem_triton.py"
    )
    @requires_triton()
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
        rank = self.rank
        # Configuration
        nreduce = 3  # number of separate reductions
        # Source buffer - each rank contributes different values
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            src[i] = (rank + 1) * (i + 1)  # Rank 0: [1,2,3], Rank 1: [2,4,6], etc.
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

        # Execute sum reduction across all ranks
        team_handle = 0  # NVSHMEM_TEAM_WORLD
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

    @unittest.skip(
        "rocSHMEM team-reduce collective symbols are unavailable in current device bitcode; see _rocshmem_triton.py"
    )
    @requires_triton()
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
        rank = self.rank
        # Configuration
        nreduce = 2  # number of values to reduce
        # Source buffers for min and max
        src_min = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        src_max = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        # Each rank contributes different values
        # For min: rank 0: [10, 20], rank 1: [15, 5], etc.
        # For max: same values
        for i in range(nreduce):
            if i == 0:
                src_min[i] = 10 + rank * 5  # 10, 15, 20, ...
                src_max[i] = 10 + rank * 5
            else:
                src_min[i] = 20 - rank * 15  # 20, 5, -10, ...
                src_max[i] = 20 - rank * 15
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
        my_reduce_kernel[(1,)](
            team_handle,
            dst_min,
            src_min,
            nreduce,
            operation="min",
            launch_cooperative_grid=True,
        )
        # Execute MAX reduction
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

    @unittest.skip(
        "rocSHMEM team-reduce collective symbols are unavailable in current device bitcode; see _rocshmem_triton.py"
    )
    @requires_triton()
    @parametrize(
        "dtype",
        [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            # torch.float64,  # Tensor-likes are not close
            torch.bfloat16,
        ],
    )
    def test_triton_prod_reduce(self, dtype) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()
        group_name = dist.distributed_c10d._get_default_group().group_name
        world_size = dist.get_world_size()
        rank = self.rank
        # Configuration
        nreduce = 3  # number of separate reductions
        # Source buffer - each rank contributes different values
        # Use very small values to avoid overflow, especially for small integer types
        src = symm_mem.empty(nreduce, dtype=dtype, device=self.device)
        for i in range(nreduce):
            # Use values that won't overflow even for int8: all values 1 or 2
            if i == 0:
                # For first element: rank 0,2,4... gets 1, rank 1,3,5... gets 2
                src[i] = 1 if rank % 2 == 0 else 2
            elif i == 1:
                # For second element: all get 1 (no multiplication effect)
                src[i] = 1
            else:
                # For third element: rank 0,1 get 1, rank 2,3 get 2, etc. (groups of 2)
                src[i] = 1 if (rank // 2) % 2 == 0 else 2
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

        # Execute product reduction across all ranks
        team_handle = 0  # NVSHMEM_TEAM_WORLD
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


if __name__ == "__main__":
    run_tests()
