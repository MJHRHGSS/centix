#ifndef ATTENTION_H
#define ATTENTION_H
#include"tensor.h"
tensor_t*attention_forward(tensor_t*q,tensor_t*k,tensor_t*v);
#endif
