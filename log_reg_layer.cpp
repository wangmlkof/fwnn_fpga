#include "log_reg_layer.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;

extern double gaussrand();

log_reg_layer::log_reg_layer(const char * layer_name,int input_size,int output_size):full_connect_layer(layer_name,input_size,output_size)
{
		initial_parameter();
		acc_flag=0;
}
void log_reg_layer::initial_parameter()
{
		cout<<"initial log layer parameter "<<endl;
		//initial W and B
		for (int i=0;i<N_out;i++)
		{
				B[i]=0;
				for(int j=0;j<N_in;j++)
				{
						W[i*N_in+j]=0;
				}
		}
}

void log_reg_layer::forward()
{
		//softmax function
		compute_sum();
		//calculate final softmax
		float sum=0;
		for(int i=0;i<N_out;i++)
		{
				out[i]=exp(Sum[i]);
				sum+=out[i];
		}
		for(int i=0;i<N_out;i++)
		{
				out[i]/=sum;
		}
}

void log_reg_layer::set_kernel_arg_forward()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(forward_kernel, 0, sizeof(int),&N_in);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 1, sizeof(cl_mem),&W_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 2, sizeof(cl_mem),&B_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 3, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 4, sizeof(cl_mem),&out_acc);
	//cout<<"set kernel arg result:"<<accelerator->err_info<<endl;	
}

void log_reg_layer::forward_acc()
{
	forward();	
/*
	accelerator->err_info=0;
	trans_in_to_acc();
	set_kernel_arg_forward();
	size_t global[1]={(size_t)(N_out)};
	accelerator->err_info|=clEnqueueNDRangeKernel(accelerator->command, forward_kernel, 1, NULL, global , NULL, 0, NULL, NULL);
	trans_out_to_host();

	clFinish(accelerator->command);
	float sum=0;
	for(int i=0;i<N_out;i++)
	{
		sum+=out[i];
	}
	for(int i=0;i<N_out;i++)
	{
		out[i]/=sum;
	}
*/
}

void log_reg_layer::backward()
{
	for(int k=0;k<N_out;k++)
	{
		for(int j=0;j<N_in;j++)
		{
			dW[k*N_in+j]+=in[j]*(-delta_in[k]);
		}
		dB[k]+=(-delta_in[k]);
	}

	for(int k=0;k<N_in;k++)
	{
		delta_out[k]=0;
		for(int m=0;m<N_out;m++)
		{
			delta_out[k]+=delta_in[m]*W[m*N_in+k];
		}	
	}
}

void log_reg_layer::backward_acc()
{
	backward();	
}
