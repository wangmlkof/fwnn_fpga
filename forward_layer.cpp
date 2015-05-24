#include "myclutils.h"
#include "forward_layer.h"
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
using namespace std;
forward_layer::forward_layer(const char * layer_name,int input_size, int output_size):name(layer_name)
{
	N_in=input_size;
	N_out=output_size;
	cout<<"construct "<<name<<" layer start"<<endl;
	//initialize all pointer to NULL
	W=dW=B=dB=Sum=in=delta_in=out=delta_out=NULL;
	//accelaration 
	accelerator=NULL;
	W_acc=B_acc=in_acc=out_acc=0;
	dW_acc=dB_acc=delta_in_acc=delta_out_acc=0;
	//profile 
	forward_time=0;
	backward_time=0;
	cout<<"construct "<<name<<" layer finish"<<endl;
}

void forward_layer::set_in(LAYER_IN_TYPE * in_p)
{
	if(in_p!=NULL)
	{
		in=in_p;
	}
	else
	{
		cout<<name<<" warnning: set input is NULL!"<<endl;
	}
}
LAYER_OUT_TYPE forward_layer::get_out(int out_index)
{
	return out[out_index];
}

void forward_layer::set_delta_in(LAYER_OUT_TYPE * in_p)
{
	if(in_p!=NULL)
	{
		delta_in=in_p;
	}
	else
	{
		cout<<"warnning: set delta input is NULL!"<<endl;
	}	
}

forward_layer::~forward_layer()
{
	cout<<"destruct "<<name<<" layer"<<endl;
	if(accelerator!=NULL)
	{
		clReleaseProgram(forward_program);
		clReleaseProgram(backward_dWB_program);
		clReleaseProgram(backward_delta_program);
		clReleaseKernel(forward_kernel);
		clReleaseKernel(backward_dWB_kernel);
		clReleaseKernel(backward_delta_kernel);
		clReleaseMemObject(W_acc);
		clReleaseMemObject(B_acc);
		clReleaseMemObject(in_acc);
		clReleaseMemObject(out_acc);
		clReleaseMemObject(dW_acc);
		clReleaseMemObject(dB_acc);
		clReleaseMemObject(delta_in_acc);
		clReleaseMemObject(delta_out_acc);
	}
	if(W!=NULL)
	{
		delete []W;
	}
	if(dW!=NULL)
	{
		delete []dW;
	}
	if(B!=NULL)
	{
		delete []B;
	}
	if(dB!=NULL)
	{
		delete []dB;
	}
	if(Sum!=NULL)
	{
		delete []Sum;
	}
	if(out!=NULL)
	{
		delete []out;
	}
	if(delta_out!=NULL)
	{
		delete []delta_out;
	}
	cout<<"destruct "<<name<<" layer finish"<<endl;
}

void forward_layer::scalar_dWB(W_TYPE scalar)
{
	for(int i=0;i<W_size;i++)
	{
		dW[i]*=scalar;
	}
	for(int i=0;i<B_size;i++)
	{
		dB[i]*=scalar;
	}
}

void forward_layer::adjust_WB()
{
	for(int i=0;i<W_size;i++)
	{
		W[i]+=dW[i];
	}
	for(int i=0;i<B_size;i++)
	{
		B[i]+=dB[i];
	}
}

void forward_layer::clean_dWB()
{
	for(int i=0;i<W_size;i++)
	{
		dW[i]=0;
	}
	for(int i=0;i<B_size;i++)
	{
		dB[i]=0;
	}
}

void forward_layer::display_out(int number)
{
	cout<<name<<" output:"<<number<<" "<<N_out<<endl;;
	if(number>=N_out||number<=0)
	{
		for(int i=0;i<N_out;i++)
		{
			//cout<<" "<<out[i];
			printf("%lf ",out[i]);
		}
	}
	else
	{
		for(int i=0;i<number;i++)
		{
			//cout<<" "<<out[i];
			printf("%lf ",out[i]);
		}
	}
	cout<<endl;
}	
void forward_layer::display_W(int number)
{
	cout<<name<<" W:"<<endl;;
	if(number>=(W_size)||number<=0)
	{
		for(int i=0;i<W_size;i++)
		{
			cout<<" "<<W[W_size];
		}
	}
	else
	{
		for(int i=0;i<number;i++)
		{
			cout<<" "<<W[i];
		}
	}
	cout<<endl;
}
void forward_layer::display_dW(int number)
{
	cout<<name<<" dW:"<<endl;;
	if(number>=(W_size)||number<=0)
	{
		for(int i=0;i<W_size;i++)
		{
			cout<<" "<<dW[W_size];
		}
	}
	else
	{
		for(int i=0;i<number;i++)
		{
			cout<<" "<<dW[i];
		}
	}
	cout<<endl;

}
void forward_layer::display_delta_out(int number)
{
	cout<<name<<" delta out:"<<number<<" "<<N_in<<endl;
	if(number>=(N_in)||number<=0)
	{
		for(int i=0;i<N_in;i++)
		{
			//cout<<" "<<delta_out[N_in];
			printf("%lf ",out[i]);
		}
	}
	else
	{
		for(int i=0;i<number;i++)
		{
			//cout<<" "<<delta_out[i];
			printf("%lf ",out[i]);
		}
	}
	cout<<endl;
}

void forward_layer::display_in(int number)
{
	cout<<name<<" in:"<<endl;
	if(number>=(N_in)||number<=0)
	{
		for(int i=0;i<N_in;i++)
		{
			cout<<" "<<in[N_in];
		}
	}
	else
	{
		for(int i=0;i<number;i++)
		{
			cout<<" "<<in[i];
		}
	}
	cout<<endl;
}

void forward_layer::clear_time_count()
{
	forward_time=0;
	backward_time=0;
}
void forward_layer::profile_time()
{
	cout.width(9);
	cout.setf(ios::fixed);
	cout<<name<<"   "<<forward_time<<"   "<<backward_time<<endl;
}
void forward_layer::scalar_time(double scalar)
{
	forward_time*=scalar;
	backward_time*=scalar;
}

void forward_layer::add_accelerator(accelerator_cl & acc)
{
	accelerator=&acc;
	prepare_acc();
}
void forward_layer::prepare_acc()
{
	if(acc_flag==1)
	{
		//related string name
		//forward
		cout<<"prepare for acceleration"<<endl;
		cout<<"read "<<name<<"kernel"<<endl;
		forward_program_name=accelerator->kernel_path+name+string("_forward.cl");
		forward_kernel_name=name+string("_forward");
		//backward dWB
		backward_dWB_program_name=accelerator->kernel_path+name+string("_backward_dWB.cl");
		backward_dWB_kernel_name=name+string("_backward_dWB");
		//backward delta
		backward_delta_program_name=accelerator->kernel_path+name+string("_backward_delta.cl");
		backward_delta_kernel_name=name+string("_backward_delta");

		//create memory buffer in accelerator
		//forward
		cout<<"create "<<name<<" accelerator memory"<<endl;
		W_acc=clCreateBuffer(accelerator->platform,CL_MEM_READ_ONLY,sizeof(W_TYPE)*W_size,NULL,NULL);
		B_acc=clCreateBuffer(accelerator->platform,CL_MEM_READ_ONLY,sizeof(B_TYPE)*B_size,NULL,NULL);
		in_acc=clCreateBuffer(accelerator->platform,CL_MEM_READ_ONLY,sizeof(LAYER_IN_TYPE)*N_in,NULL,NULL);
		out_acc=clCreateBuffer(accelerator->platform,CL_MEM_WRITE_ONLY,sizeof(LAYER_OUT_TYPE)*N_out,NULL,NULL);
		//backward
		dW_acc=clCreateBuffer(accelerator->platform,CL_MEM_WRITE_ONLY,sizeof(W_TYPE)*W_size,NULL,NULL);
		dB_acc=clCreateBuffer(accelerator->platform,CL_MEM_WRITE_ONLY,sizeof(B_TYPE)*B_size,NULL,NULL);
		delta_in_acc=clCreateBuffer(accelerator->platform,CL_MEM_READ_ONLY,sizeof(LAYER_IN_TYPE)*N_out,NULL,NULL);
		delta_out_acc=clCreateBuffer(accelerator->platform,CL_MEM_WRITE_ONLY,sizeof(LAYER_OUT_TYPE)*N_in,NULL,NULL);

		if (!W_acc||!B_acc||!in_acc||!out_acc||!dW_acc||!dB_acc||!delta_in_acc ||!delta_out_acc)
		{
			cout<<"Error: Failed to allocate device memory!"<<endl;
		}
		//create  kenerl and program
		cout<<"create "<<name<<" opencl program"<<endl;
		//forward
		//forward_program=CreateProgram_bin(accelerator->platform,accelerator->device,accelerator->xclbin_name.c_str());
		forward_kernel=clCreateKernel(accelerator->program,forward_kernel_name.c_str(),&(accelerator->err_info));
		if (!forward_kernel|| accelerator->err_info != CL_SUCCESS)
		{
			cout<<"Error: Failed to create compute kernel: "<<forward_kernel_name<<endl;
			cout<<getErrorString(accelerator->err_info)<<endl;
		}

		//bakcword
		//dWB
		//backward_dWB_program=CreateProgram_bin(accelerator->platform,accelerator->device,accelerator->xclbin_name.c_str());
		backward_dWB_kernel=clCreateKernel(accelerator->program,backward_dWB_kernel_name.c_str(),&(accelerator->err_info));
		if (!backward_dWB_kernel|| accelerator->err_info != CL_SUCCESS)
		{
			cout<<"Error: Failed to create compute kernel: "<<backward_dWB_kernel_name<<endl;
			cout<<getErrorString(accelerator->err_info)<<endl;
		}
		//delta
		//backward_delta_program=CreateProgram_bin(accelerator->platform,accelerator->device,accelerator->xclbin_name.c_str());
		backward_delta_kernel=clCreateKernel(accelerator->program,backward_delta_kernel_name.c_str(),&(accelerator->err_info));
		if (!backward_delta_kernel|| accelerator->err_info != CL_SUCCESS)
		{
			cout<<"Error: Failed to create compute kernel: "<<backward_delta_kernel_name<<endl;
			cout<<getErrorString(accelerator->err_info)<<endl;
		}
	}
}

void forward_layer::trans_WB_to_acc()
{
	accelerator->err_info=0;
	if(W_size>0)
	{
		accelerator->err_info|=clEnqueueWriteBuffer(accelerator->command, W_acc, CL_TRUE, 0,sizeof(W_TYPE) * W_size, W, 0, NULL, NULL);
	}
	if(B_size>0)
	{
		accelerator->err_info|=clEnqueueWriteBuffer(accelerator->command, B_acc, CL_TRUE, 0,sizeof(B_TYPE) * B_size, B, 0, NULL, NULL);
	}
	//	cout<<name<<" transfer W B to accelerator result:"<<accelerator->err_info<<endl;	
}

void forward_layer::trans_dWB_to_acc()
{
	accelerator->err_info=0;
	if(W_size>0)
	{
		accelerator->err_info|=clEnqueueWriteBuffer(accelerator->command, dW_acc, CL_TRUE, 0,sizeof(W_TYPE) * W_size, dW, 0, NULL, NULL);
	}
	if(B_size>0)
	{
		accelerator->err_info|=clEnqueueWriteBuffer(accelerator->command, dB_acc, CL_TRUE, 0,sizeof(B_TYPE) * B_size, dB, 0, NULL, NULL);
	}
	//	cout<<name<<" transfer dWB to accelerator result:"<<accelerator->err_info<<endl;	
}

void forward_layer::trans_in_to_acc()
{
	accelerator->err_info=0;
	accelerator->err_info|=clEnqueueWriteBuffer(accelerator->command, in_acc, CL_TRUE, 0,sizeof(LAYER_IN_TYPE) * N_in, in, 0, NULL, NULL);
	//	cout<<name<<"transfer in["<<N_in<<"]to accelerator result:"<<accelerator->err_info<<endl;	
}

void forward_layer::trans_delta_in_to_acc()
{
	accelerator->err_info=0;
	accelerator->err_info|=clEnqueueWriteBuffer(accelerator->command, delta_in_acc, CL_TRUE, 0,sizeof(LAYER_IN_TYPE) * N_out, delta_in, 0, NULL, NULL);
	//	cout<<name<<"transfer in["<<N_in<<"]to accelerator result:"<<accelerator->err_info<<endl;	
}

void forward_layer::trans_out_to_host()
{
	accelerator->err_info=clEnqueueReadBuffer(accelerator->command, out_acc,CL_TRUE, 0, sizeof(LAYER_OUT_TYPE)*N_out,out, 0, NULL, NULL);
}

void forward_layer::trans_delta_out_to_host()
{
	accelerator->err_info=clEnqueueReadBuffer(accelerator->command, delta_out_acc,CL_TRUE, 0, sizeof(LAYER_OUT_TYPE)*N_in,delta_out, 0, NULL, NULL);
	//	cout<<name<<"transfer out["<<N_out<<"]to accelerator result:"<<accelerator->err_info<<endl;	

}

void forward_layer::trans_dWB_to_host()
{
	
	accelerator->err_info=0;
	if(W_size>0)
	{ 
		accelerator->err_info|=clEnqueueReadBuffer(accelerator->command, dW_acc,CL_TRUE, 0, sizeof(W_TYPE)*W_size,dW, 0, NULL, NULL);
	}
	if(B_size>0)
	{
		accelerator->err_info|=clEnqueueReadBuffer(accelerator->command, dB_acc,CL_TRUE, 0, sizeof(B_TYPE)*B_size,dB, 0, NULL, NULL);
	}
	//	cout<<name<<"transfer out["<<N_out<<"]to accelerator result:"<<accelerator->err_info<<endl;	
}

