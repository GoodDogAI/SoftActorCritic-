#include "normal.h"

Normal::Normal(Tensor mean, Tensor standardDeviation)
    : mean(mean), stddev(standardDeviation) { }

Tensor _standard_normal(c10::IntArrayRef size, const c10::TensorOptions& options) {
    return torch::empty(size, options).normal_();
};

Tensor Normal::rsample()
{
    auto eps = _standard_normal(this->mean.sizes(), this->mean.options());
    return this->mean + eps * this->stddev;
}

Tensor Normal::logProb(Tensor value)
{
    auto variance = torch::pow(this->stddev, 2);
    auto log_scale = this->stddev.log();
    return -torch::pow(value - this->mean, 2) / (2 * variance) - log_scale - log(sqrt(2 * M_PI));
}
