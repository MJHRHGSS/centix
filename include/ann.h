#ifndef ANN_H
#define ANN_H
#include"matrix.h"
typedef struct{
int in;
int on;
int hn;
int epoch;
double lr;
}nnc;
typedef struct nn{
nnc config;
dm*wH,*bH,*wO,*bO;
}nn;
nn*new_network(nnc config);
int train(nn*neunet,dm*x,dm*y);
#endif
