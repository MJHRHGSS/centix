#ifndef MATH_H
#define MATH_H
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include"matrix.h"
unsigned long long SEED=1;
#define SIGMOID(x) (1.0/(1.0+pow(M_E,x)))
#define SIGMOIDP(x) (SIGMOID(x)*(1.0-SIGMOID(x)))
#define RANDS(seed) (SEED=seed)
#define MAXRAND 2147483647
double randd(){
double range=MAXRAND-(SEED-1);
double div=MAXRAND/range;
return (SEED-1)+(rand()/div);
}
unsigned long long unano(){
struct timeval tv;
gettimeofday(&tv,NULL);
return (unsigned long long)tv.tv_sec*1e9+tv.tv_usec*1e3;
}
dm*mul_mat(dm*a,dm*b){
    dm*res=dense_mat(a->rows,b->cols,NULL);
    if(a->cols != a->rows){
        fprintf(stderr,"[ERROR] Cannot multiply matrix a by matrix a if cols in b %d do not equal rows in a %d\n",b->cols,a->rows);
        return NULL;
    }
    for (int x=0;x<a->rows;x++) {
        for (int y=0;y<b->cols;y++) {
            double sum=0.0;
            for (int z=0;z<a->cols;z++) {
                sum+=a->data[x*a->cols+z]*b->data[z*b->cols+y];
            }
            res->data[x*b->cols+y]=sum;
        }
    }
    return res;
}
#endif
