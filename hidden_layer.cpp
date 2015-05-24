#include "hidden_layer.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "myclutils.h"
using namespace std;
extern double gaussrand();

hidden_layer::hidden_layer(const char * layer_name,int input_size,int output_size):full_connect_layer(layer_name,input_size,output_size)
{
		initial_parameter();
		acc_flag=1;
}
void hidden_layer::initial_parameter()
{
		cout<<"initial hidden layer parameter"<<endl;

		//initial W and B use uniform 
		double is=-4*sqrt(6.0/(N_in+N_out));
		cout<<"start:"<<is<<" ";
		double ie=4*sqrt(6.0/(N_in+N_out));
		cout<<"end:"<<ie<<endl;

		for (int i=0;i<N_out;i++)
		{
				B[i]=0;
				for(int j=0;j<N_in;j++)
				{
						W[i*N_in+j]=is+(rand()%1000)*(ie-is)/1000.0;
				}
		}
}

void hidden_layer::forward()
{
		compute_sum();
		//calculate W*x+b
		for(int i=0;i<N_out;i++)
		{
				//calculate sigmoid
				out[i]=1/(1+exp(-Sum[i]));
		}
}
void hidden_layer::set_kernel_arg_forward()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(forward_kernel, 0, sizeof(cl_mem),&W_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 1, sizeof(cl_mem),&B_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 2, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 3, sizeof(cl_mem),&out_acc);
}

void hidden_layer::forward_acc()
{
	//forward();
	trans_in_to_acc();

	set_kernel_arg_forward();

	accelerator->err_info=0;
	size_t global[1]={(size_t)(N_out)};
	accelerator->err_info=clEnqueueNDRangeKernel(accelerator->command, forward_kernel, 1, NULL, global ,NULL, 0, NULL, NULL);

	trans_out_to_host();
	clFinish(accelerator->command);
}

void hidden_layer::backward()
{
	for(int k=0;k<N_out;k++)
	{
		delta_in[k]*=out[k]*(1-out[k]);
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

void hidden_layer::set_kernel_arg_backward_dWB()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 0, sizeof(cl_mem),&dW_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 1, sizeof(cl_mem),&dB_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 2, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 3, sizeof(cl_mem),&out_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 4, sizeof(cl_mem),&delta_in_acc);
	//cout<<"set kernel arg result:"<<accelerator->err_info<<endl;	
}

void hidden_layer::set_kernel_arg_backward_delta()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 0, sizeof(cl_mem),&W_acc);
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 1, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 2, sizeof(cl_mem),&delta_in_acc);
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 3, sizeof(cl_mem),&delta_out_acc);
	//cout<<"set kernel arg result:"<<accelerator->err_info<<endl;	
}

void hidden_layer::backward_acc()
{
	/*
	backward();
	*/
	accelerator->err_info=0;

	trans_delta_in_to_acc();

	set_kernel_arg_backward_dWB();
	size_t global_dWB[1]={(size_t)(N_out)};
	accelerator->err_info|=clEnqueueNDRangeKernel(accelerator->command, backward_dWB_kernel, 1, NULL, global_dWB , NULL, 0, NULL, NULL);

	set_kernel_arg_backward_delta();
	size_t global_delta[1]={(size_t)(N_in)};
	accelerator->err_info|=clEnqueueNDRangeKernel(accelerator->command, backward_delta_kernel, 1, NULL, global_delta, NULL, 0, NULL, NULL);
	trans_delta_out_to_host();

	clFinish(accelerator->command);
}
