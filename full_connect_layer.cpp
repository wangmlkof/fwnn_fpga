#include "full_connect_layer.h"
#include<iostream>
#include <stdlib.h>
using namespace std;
full_connect_layer::full_connect_layer(const char * layer_name,int input_size, int output_size):forward_layer(layer_name,input_size,output_size)
{
		W_size=N_in*N_out;
		B_size=N_out;
		alloc_parameter();
}

void full_connect_layer::compute_sum()
{
		for(int i=0;i<N_out;i++)
		{
				Sum[i]=0;
				for(int j=0;j<N_in;j++)
				{
						Sum[i]+=W[i*N_in+j]*in[j];
				}
				Sum[i]+=B[i];
		}
}
void full_connect_layer::alloc_parameter()
{
		cout<<"start allocating "<<name<<" parameter space"<<endl;
		//allocate space for parameter of layer
		W=new W_TYPE[N_in*N_out];
		dW=new W_TYPE[N_in*N_out];
		B=new B_TYPE[N_out];
		dB=new B_TYPE[N_out];
		Sum=new W_TYPE[N_out];
		out=new LAYER_OUT_TYPE[N_out];
		delta_out=new LAYER_OUT_TYPE[N_in];
		if(W==NULL||dW==NULL||B==NULL||dB==NULL||Sum==NULL||out==NULL||delta_out==NULL)
		{
				cout<<"alloc space failed"<<endl;
				exit(1);
		}
}

