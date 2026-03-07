#pragma once
enum class GemmType {
    Normal = 0,
    MGroupedContiguous, // each expert's okens are concatened into a single tensor and passed in
    MGroupedMasked, // used in decoding phase when CUDa graph is enabled and CPU doesn't know the number of tokens received, so set a max and provide a mask tensor
    Batched,
};

enum class Major {
    MN,
    K
};

enum class KernelType {
    Vanilla = 0,
    Scaled1D2D,
};

enum class ScaleFactor {
    None = 0,
    Scale1D_128,
    Scale2D_128
};

enum InferenceMode {
    Prefill = 0,
    Decode,
    SpecVerify,
    SpecDecode
};