#include "conv_layer.h"
#include<math.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

extern double gaussrand();
conv_layer::conv_layer(const char * layer_name,int mapinnum,int mapinsidelen,int mapoutnum,int mapoutsidelen,int kernelsidelen):
	forward_layer(layer_name,mapinnum*mapinsidelen*mapinsidelen,mapoutnum*mapoutsidelen*mapoutsidelen),
	map_in_num(mapinnum),
	map_in_side_len(mapinsidelen),
	map_out_num(mapoutnum),
	map_out_side_len(mapoutsidelen),
	kernel_side_len(kernelsidelen)
{
	cout<<"construct conv layer:"<<endl;
	cout<<"input map: "<<map_in_num<<","<<map_in_side_len<<endl;
	cout<<"output map: "<<map_out_num<<","<<map_out_side_len<<endl;
	W_size=map_out_num*map_in_num*kernel_side_len*kernel_side_len;
	B_size=map_out_num;
	alloc_parameter();
	initial_parameter();

	acc_flag=1;
}

void conv_layer::alloc_parameter()
{
	cout<<"start allocating "<<name<<" parameter space"<<endl;
	//allocate space for parameter of layer
	W=new W_TYPE[W_size];
	dW=new W_TYPE[W_size];
	B=new B_TYPE[B_size];
	dB=new B_TYPE[B_size];

	Sum=new W_TYPE[N_out];
	out=new LAYER_OUT_TYPE[N_out];
	delta_out=new LAYER_OUT_TYPE[N_in];
	if(W==NULL||dW==NULL||B==NULL||dB==NULL||Sum==NULL||out==NULL||delta_out==NULL)
	{
		cout<<"alloc space failed"<<endl;
		exit(1);
	}

}

void conv_layer::initial_parameter()
{
	for(int i=0;i<W_size;i++)
	{
		W[i]=gaussrand();
	}
	for(int i=0;i<B_size;i++)
	{
		B[i]=0;
	}
}
void conv_layer::forward()
{
	for(int l=0;l<map_out_num;l++)
	{
		for(int m=0;m<map_out_side_len;m++)
		{
			for(int n=0;n<map_out_side_len;n++)
			{
				int sum_index=l*map_out_side_len*map_out_side_len+m*map_out_side_len+n;
				Sum[sum_index]=0;
				for(int i=0;i<map_in_num;i++)
				{
					for(int j=0;j<kernel_side_len;j++)
					{
						for(int k=0;k<kernel_side_len;k++)
						{
							Sum[sum_index]+=W[l*map_in_num*kernel_side_len*kernel_side_len+i*kernel_side_len*kernel_side_len+j*kernel_side_len+k]*in[i*map_in_side_len*map_in_side_len+(m+j)*map_in_side_len+(n+k)];
						}
					}
				}
				Sum[sum_index]+=B[l];
				out[sum_index]=1/(1+exp(-Sum[sum_index]));
			}
		}
	}
}

void conv_layer::set_kernel_arg_forward()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(forward_kernel, 0, sizeof(cl_mem),&W_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 1, sizeof(cl_mem),&B_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 2, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(forward_kernel, 3, sizeof(cl_mem),&out_acc);
	//cout<<"set kernel arg result:"<<accelerator->err_info<<endl;	
}

void conv_layer::forward_acc()
{
	accelerator->err_info=0;
	trans_in_to_acc();
	set_kernel_arg_forward();
	size_t global[3]={(size_t)map_out_num,(size_t)map_out_side_len,(size_t)map_out_side_len};
	accelerator->err_info=clEnqueueNDRangeKernel(accelerator->command, forward_kernel, 3, NULL, global , NULL, 0, NULL, NULL);
	trans_out_to_host();
	clFinish(accelerator->command);
}

void conv_layer::backward()
{
	//adjust W and B according to delta in
	for(int l=0;l<map_out_num;l++)
	{
		for(int m=0;m<map_out_side_len;m++)
		{
			for(int n=0;n<map_out_side_len;n++)
			{
				int out_index=l*map_out_side_len*map_out_side_len+m*map_out_side_len+n;
				delta_in[out_index]*=out[out_index]*(1-out[out_index]);
				for(int i=0;i<map_in_num;i++)
				{
					for(int j=0;j<kernel_side_len;j++)
					{
						for(int k=0;k<kernel_side_len;k++)
						{
							dW[l*map_in_num*kernel_side_len*kernel_side_len+i*kernel_side_len*kernel_side_len+j*kernel_side_len+k]+=in[i*map_in_side_len*map_in_side_len+(m+j)*map_in_side_len+(n+k)]*(-delta_in[out_index]);
						}
					}
				}
				dB[out_index]+=(-delta_in[out_index]);
			}
		}
	}
	//backward delta
	for(int i=0;i<map_in_num;i++)
	{
		for(int j=0;j<kernel_side_len;j++)
		{
			for(int k=0;k<kernel_side_len;k++)
			{
				for(int m=0;m<map_out_side_len;m++)
				{
					for(int n=0;n<map_out_side_len;n++)
					{
						int in_index=i*map_in_side_len*map_in_side_len+(m+j)*map_in_side_len+(n+k);
						delta_out[in_index]=0;
						for(int l=0;l<map_out_num;l++)
						{
							W_TYPE w_temp=W[l*map_in_num*kernel_side_len*kernel_side_len+i*kernel_side_len*kernel_side_len+j*kernel_side_len+k];
							int out_index=l*map_out_side_len*map_out_side_len+m*map_out_side_len+n;
							delta_out[in_index]+=delta_in[out_index]*w_temp;
						}
					}
				}
			}
		}
	}
}

void conv_layer::set_kernel_arg_backward_dWB()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 0, sizeof(cl_mem),&dW_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 1, sizeof(cl_mem),&dB_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 2, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 3, sizeof(cl_mem),&out_acc);
	accelerator->err_info|=clSetKernelArg(backward_dWB_kernel, 4, sizeof(cl_mem),&delta_in_acc);
}

void conv_layer::set_kernel_arg_backward_delta()
{
	accelerator->err_info=0;
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 0, sizeof(cl_mem),&W_acc);
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 1, sizeof(cl_mem),&in_acc);
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 2, sizeof(cl_mem),&delta_in_acc);
	accelerator->err_info|=clSetKernelArg(backward_delta_kernel, 3, sizeof(cl_mem),&delta_out_acc);
}

void conv_layer::backward_acc()
{
	/*
	accelerator->err_info=0;

	trans_delta_in_to_acc();

	set_kernel_arg_backward_dWB();
	size_t global_dWB[3]={(size_t)map_out_num,(size_t)map_out_side_len,(size_t)map_out_side_len};
	accelerator->err_info|=clEnqueueNDRangeKernel(accelerator->command, backward_dWB_kernel, 3, NULL, global_dWB , NULL, 0, NULL, NULL);

	set_kernel_arg_backward_delta();
	size_t global_delta[3]={(size_t)map_in_num,(size_t)map_out_side_len,(size_t)map_out_side_len};
	accelerator->err_info|=clEnqueueNDRangeKernel(accelerator->command, backward_delta_kernel, 3, NULL, global_delta, NULL, 0, NULL, NULL);

	trans_delta_out_to_host();

	clFinish(accelerator->command);
	*/
	backward();
}
