#include <torch/torch.h>

using namespace torch;

class Normal {
    Tensor mean, stddev;

public:
    Normal(Tensor mean, Tensor standardDeviation);
    Tensor rsample();
    Tensor logProb(Tensor value);
};