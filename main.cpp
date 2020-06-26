#include <torch/torch.h>
#include <iostream>

#include "sac.h"

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
