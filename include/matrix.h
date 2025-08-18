#ifndef MATRIX_H
#define MATRIX_H
#include<stdio.h>
#include<stdlib.h>
typedef struct{
int rows,cols;
double*data;
}dm;
dm*dense_mat(int r,int c,double data[]){
dm*m=malloc(sizeof(dm));
if(data==NULL)m->data=malloc((size_t)(r*c));
else m->data=malloc(sizeof(data)/sizeof(data[0]));
m->cols=c;
m->rows=r;
return m;
}
void free_dm(dm*m){free(m);}
#endif
