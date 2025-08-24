#include "model.h"
#include "attention.h"
#include <stdlib.h>
tensor_t*run_dummy_model(size_t seq_len,size_t dim) {
tensor_t*q=tensor_alloc(seq_len,dim);
tensor_t*k=tensor_alloc(seq_len,dim);
tensor_t*v=tensor_alloc(seq_len,dim);
tensor_fill_random(q);
tensor_fill_random(k);
tensor_fill_random(v);
tensor_t*out=attention_forward(q,k,v);
tensor_free(q);
tensor_free(k);
tensor_free(v);
return out;
}
