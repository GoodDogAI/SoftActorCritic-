#include "sac.h"

template<class Inner>
SquashedGaussianMLPActor<Inner>::SquashedGaussianMLPActor(std::shared_ptr<Inner> inner, int actionDim, float actionLimit)
{
    this->inner = inner;
    int innerOutputSize = 10;
    this->mu = torch::nn::Linear(innerOutputSize, actionDim);
    this->logStd = torch::nn::Linear(innerOutputSize, actionDim);
};

const float LOG_STD_MAX = 2;
const float LOG_STD_MIN = -20;

template<class Inner>
torch::Tensor SquashedGaussianMLPActor<Inner>::forward(torch::Tensor input, bool deterministic, Tensor& logProb)
{
    Tensor innerOut = this->inner->forward(input);
    Tensor mu = this->mu->forward(innerOut);
    Tensor logStd = clamp(this->logStd->forward(innerOut), LOG_STD_MIN, LOG_STD_MAX);
    Tensor std = exp(logStd);
    auto pi_distribution = torch::distributions normal(mu, std);
    Tensor pi_action = deterministic ? mu : pi_distribution.rsample();
    if (logProb) {
        logProb = pi_distribution.log_prob(pi_action).sum(axis=-1);
        logProb -= (2 * (log(2f) - pi_action - softplus(-2 * pi_action))).sum(axis = 1)
    }
    pi_action = tanh(pi_action);
    pi_action = this.actionLimit * pi_action;
    return pi_action;
};
