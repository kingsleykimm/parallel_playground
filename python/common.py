import torch

def quantize_1d_128(input : torch.Tensor):
    assert input.dtype == torch.float or input.dtype == torch.bfloat16
    flattened = input.reshape(-1, input.size(-1))
    num_blocks = input.size(-1) / 128

    quantized = torch.empty_like(flattened, dtype = torch.float8_e4m3fn, device = flattened.device)
    scales = torch.empty(flattened.size(0), num_blocks, dtype = torch.float32, device = flattened.device)
    for i in range(flattened.size(0)):

        for block in range(num_blocks):
            slice = flattened[i, block * 128 : (block + 1) * 128]
            cur_scale = slice.amax() / 448.0
            scales[i, block] = cur_scale
            quantized[i, block * 128 : (block + 1) * 128] = (slice / cur_scale).to(torch.float8_e4m3fn)
    return quantized, scales

def quantize_2d_128(input : torch.Tensor):
    assert input.dtype == torch.float or input.dtype == torch.bfloat16
    assert input.size(-1) % 128 == 0 and input.size(-2) % 128 == 0
    num_groups = input.size(0)
    num_k_blocks = input.size(-1) / 128
    num_n_blocks = input.size(-2) / 128

    quantized = torch.empty_like(input, dtype = torch.float8_e4m3fn, device = input.device)
    scales = torch.empty(num_groups, num_n_blocks, num_k_blocks, dtype = torch.float32, device = input.device)

    for g in range(num_groups):
        for n_block in range(num_n_blocks):
            for k_block in range(num_k_blocks):
                slice2d = input[g, n_block * 128 : (n_block + 1) * 128, k_block * 128 : (k_block + 1) * 128]
                cur_scale = slice2d.amax() / 448.0
                scales[g, n_block, k_block] = cur_scale
                quantized[g, n_block * 128 : (n_block + 1) * 128, k_block * 128 : (k_block + 1) * 128] = (slice2d / cur_scale).to(torch.float8_e4m3fn)
    return quantized, scales