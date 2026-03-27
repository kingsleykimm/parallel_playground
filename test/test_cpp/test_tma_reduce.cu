#include <kittens.cuh>

constexpr int NUM_DEVICES = 4;

using namespace kittens;
struct globals {

  using out_tokens_layout = pgl<gl<bf16, -1, -1, 1, 1>, 4, false>;
  using expert_y_layout = gl <
}