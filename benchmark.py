import torch
import triton
import triton.language as tl
import timeit
from triton.runtime import driver
from torch.utils.cpp_extension import load
torch.set_default_device('cuda')


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
def softmax_triton(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 1
    y = torch.empty_like(x)
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,)
    return y

def sync_wrapper(callable):
    out = callable()
    torch.cuda.synchronize()

def time_fn(callable, reps):
    return timeit.timeit(lambda: sync_wrapper(callable), number=reps)


x = torch.rand(128, 2**17, device='cuda')

y = torch.softmax(x, dim=-1)
y2 = softmax_triton(x)
reps = 400
  
torch_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1), rep=reps)
torch_ms_timeit = time_fn(lambda: torch.softmax(x, dim=-1), reps=reps)
print(f"torch function, triton reported {torch_ms:.4f}, timeit reported {torch_ms_timeit:.4f}")

triton_ms = triton.testing.do_bench(lambda: softmax_triton(x), rep=reps)
triton_ms_timeit = time_fn(lambda: softmax_triton(x), reps=reps)

print(f"triton function, triton reported {triton_ms:.4f}, timeit reported {triton_ms_timeit:.4f}")



for variant in range(1, 9):
    cuda = load(name='softmax_cuda', sources=["interface.cpp", "kernels.cu"], verbose=False, extra_cuda_cflags=[f"-lineinfo", "--use_fast_math", "-O3", f"-DSOFTMAX_VARIANT={variant}" ])
    y3 = cuda.softmax_cuda(x)

    assert torch.allclose(y, y2, atol=1e-6, rtol=1e-6), (y, y2)
    assert torch.allclose(y, y3, atol=1e-6, rtol=1e-6), (y, y3)
    cuda_ms = triton.testing.do_bench(lambda: cuda.softmax_cuda(x), rep=reps)
    cuda_ms_timeit = time_fn(lambda: cuda.softmax_cuda(x), reps=reps)
    print(f"variant {variant}, triton reported {cuda_ms:.4f}, timeit reported {cuda_ms_timeit:.4f}")

