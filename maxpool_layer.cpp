#include "maxpool_layer.h"
#include<stdlib.h>
#include<iostream>
#include<stdio.h>
#include<math.h>
using namespace std;

maxpool_layer::maxpool_layer(const char * layer_name,int input_size, int output_size):
	forward_layer(layer_name,input_size,output_size),
	pool_size(input_size/output_size)
{
	cout<<"construct max pooling layer"<<endl;
	cout<<"pool size: "<<pool_size<<endl;
	pool_side_len=sqrt(pool_size);
	cout<<"pool side len: "<<pool_side_len<<endl;
	W_size=0;
	B_size=0;	
	alloc_parameter();
	acc_flag=0;
}
void maxpool_layer::alloc_parameter()
{
	cout<<"start allocating "<<name<<" parameter space"<<endl;

	out=new LAYER_OUT_TYPE[N_out];
	delta_out=new LAYER_OUT_TYPE[N_in];
	if(out==NULL||delta_out==NULL)
	{
		cout<<"alloc space failed"<<endl;
		exit(1);
	}
}
void maxpool_layer::initial_parameter()
{

}
void maxpool_layer::forward()
{
	for(int i=0;i<N_out;i++)
	{
		LAYER_OUT_TYPE max_value=in[i*pool_size];
		for(int j=0;j<pool_side_len;j++)
		{
			for(int k=0;k<pool_side_len;k++)		
			{
				if(in[i*pool_size+j*pool_side_len+k]>max_value)
				{
					max_value=in[i*pool_size+j*pool_side_len+k];
				}
			}
		}
		out[i]=max_value;
	}
}
void maxpool_layer::set_kernel_arg_forward()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(forward_kernel, 0, sizeof(int),&pool_side_len);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 1, sizeof(int),&pool_size);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 2, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 3, sizeof(cl_mem),&out_acc);
	//cout<<"set kernel arg result:"<<accelerator->err_info<<endl;	
}

void maxpool_layer::forward_acc()
{
	forward();
/*
	accelerator->err_info=0;
	trans_in_to_acc();
	set_kernel_arg_forward();
	size_t global[1]={(size_t)N_out};
	accelerator->err_info|=clEnqueueNDRangeKernel(accelerator->command, forward_kernel, 1, NULL, global , NULL, 0, NULL, NULL);
	trans_out_to_host();
	clFinish(accelerator->command);
	cout<<name<<"forward acc result:"<<accelerator->err_info<<endl;
*/
}
void maxpool_layer::backward()
{
	for(int i=0;i<N_out;i++)
	{
		int max_x=0;
		int max_y=0;
		LAYER_OUT_TYPE max_value=in[i*pool_size];
		for(int j=0;j<pool_side_len;j++)
		{
			for(int k=0;k<pool_side_len;k++)		
			{
				delta_out[i*pool_size+j*pool_side_len+k]=0;
				if(in[i*pool_size+j*pool_side_len+k]>max_value)
				{
					max_x=j;
					max_y=k;
					max_value=in[i*pool_size+j*pool_side_len+k];
				}
			}
		}
		delta_out[i*pool_size+max_x*pool_side_len+max_y]=delta_in[i];
	}
}
void maxpool_layer::backward_acc()
{
	backward();
}
