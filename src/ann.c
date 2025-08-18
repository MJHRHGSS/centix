#include<stdlib.h>
#include<math.h>
#include"ann.h"
#include"matrix.h"
#include"maths.h"
int bp(nn*this,dm*x,dm*y,dm*wH,dm*bH,dm*wO,dm*bO,dm*out){
for(int n=0;n<this->config.epoch;n++){
dm*hli=dense_mat(0,0,NULL);

}
}
nn*new_network(nnc config){return &(nn){config,&(dm){},&(dm){},&(dm){},&(dm){}};}
int train(nn*this,dm*x,dm*y){
srand(unano());
dm*wH=&(dm){this->config.in,this->config.hn,NULL};
dm*bH=&(dm){1,this->config.hn,NULL};
dm*wO=&(dm){this->config.hn,this->config.on,NULL};
dm*bO=&(dm){1,this->config.on,NULL};
double*wHr=wH->data;
double*bHr=bH->data;
double*wOr=wO->data;
double*bOr=bO->data;
double*arrs[]={wHr,bHr,wOr,bOr};
for(int x=0;x<4;x++)for(int y=0;y<sizeof(arrs[x])/sizeof(arrs[x][0]);y++)arrs[x][y]=randd();
dm*output=dense_mat(0,0,NULL);
return 1;
}
