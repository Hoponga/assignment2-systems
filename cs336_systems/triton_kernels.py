import torch
import triton
import triton.language as tl
from triton.runtime import driver 
from einops import rearrange

DEVICE = torch.device("cuda")

properties = driver.active.utils.get_device_properties(torch.cuda.current_device())
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


print(f"GPU Properties: ")
print(f"Number of SMs: {NUM_SM}")
print(f"Max reg count: {NUM_REGS}")
print(f"Smem size: {SIZE_SMEM}")
print(f"target: {target}")

@triton.jit 
def weighted_sum_fwd(
    x_ptr, weight_ptr, output_ptr, x_stride_row, x_stride_dim, 
    weight_stride_dim, output_stride_row, ROWS, D, ROWS_TILE_SIZE, D_TILE_SIZE): 
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        x_ptr, 
        shape = (ROWS, D,), 
        strides = (x_row_stride, x_stride_dim), 
        offsets = (row_tile_idx * ROWS_TILE_SIZE, 0), 
        block_shape = (ROWS_TILE_SIZE, D_TILE_SIZE), 
        order = (1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, 
        shape = (D,), 
        strides = (weight_stride_dim,),
        offsets = (0,), 
        block_shape = (D_TILE_SIZE,), 
        order = (0,)
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape = (ROWS,),
        strides = (output_stride_row,),
        offsets = (row_tile_idx * ROWS_TILE_SIZE),
        block_shape = (ROWS_TILE_SIZE,), 
        order = (0,)
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype = tl.float32) 

    for i in range(tl.cdiv(D, D_TILE_SIZE)): 
        row = tl.load(x_block_ptr, boundary_check = (0, 1), padding_option = "zero")
        weight = tl.load(weight_block_ptr, boundary_check = (0,), padding_option = "zero")


        output += tl.sum(row * weight[None, :], axis = 1)


        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))

        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) 

    tl.store(output_block_ptr, output, boundary_check = (0,))

class WeightedSumFunc(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, x, weight): 
        D, output_dims = x.shape[-1], x.shape[:-1]

        input_shape = x.shape 
        x = rearrange(x, " ... d -> (...) d") # make (B, S, num_heads, D) to (..., D)

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"

        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"

        assert x.is_contiguous(), "pointer arithmetic assumes contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)

        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device = x.device)

        n_rows = y.numel() 

        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, 
            y, x.stride(0), x.stride(1), 
            weight.stride(0), y.stride(0), 
            ROWS = n_rows, D = D, 
            ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE, D_TILE_SIZE = ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])
        




def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]


    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


# can speed up softmax via triton by bringing chunks into sram and doing full computation on them 
@triton.jit 
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr): 
    row_start = tl.program_id(0) 
    row_step = tl.num_programs(0)


    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages): 

        row_start_ptr = input_ptr + row_idx * input_row_stride # input_row_stride is "memory size" of one row 
        col_offsets = tl.arange(0, BLOCK_SIZE) # one row will fit in single block 



        input_ptrs = row_start_ptr + col_offsets # [row_start_ptr, row_start_ptr + 1, ..., row_start_ptr + BLOCK_SIZE - 1]

        mask = col_offsets < n_cols
        # each row is loaded into SRAM only ONCE 
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets

        # each row is stored back into SRAM only ONCE 
        tl.store(output_ptrs, softmax_output, mask=mask)




def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols)
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)






if __name__ == "__main__": 
    # test out basic correctness 

    benchmark.run(show_plots=True, print_data=True)


