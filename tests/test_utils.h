#pragma once
#include "Tensor.h"

void fill_sequential(Tensor &m);
float square(float s);
Tensor naive_matmul(const Tensor &A, const Tensor &B);