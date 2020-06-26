#include <torch/torch.h>

using namespace torch;

template <class Inner> struct SquashedGaussianMLPActor : torch::nn::Module {
    std::shared_ptr<Inner> inner;
    torch::nn::Linear mu, logStd;

    SquashedGaussianMLPActor(std::shared_ptr<Inner> inner, int actionDim, float actionLimit);
    torch::Tensor forward(torch::Tensor input, bool deterministic, Tensor& logProb);
};