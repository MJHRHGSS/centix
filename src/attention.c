#include"attention.h"
#include<math.h>
#include<stdlib.h>
tensor_t*attention_forward(tensor_t*q,tensor_t*k,tensor_t*v){
size_t seq_len=q->rows;
size_t dim=q->cols;
tensor_t*out=tensor_alloc(seq_len,dim);
if(!out)return NULL;
for(size_t i=0;i<seq_len;i++){
for(size_t j=0;j<dim;j++){
float sum=0.0f;
for (size_t t=0;t<seq_len;t++){
float dot=0.0f;
for (size_t d=0;d<dim;d++){
dot+=q->data[i*dim+d]*k->data[t*dim+d];
}
sum+=dot*v->data[t*dim+j];
}
out->data[i*dim+j]=sum;
}
}
return out;
}
