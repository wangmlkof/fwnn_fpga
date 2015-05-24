#ifndef FORWARD_LAYERB
#define FORWARD_LAYERB
#include<string>
#include "accelerator_cl.h"
typedef float W_TYPE;
typedef float B_TYPE;

typedef float LAYER_OUT_TYPE;
typedef float LAYER_IN_TYPE;

using namespace std;

class forward_layer
{
	protected:
		W_TYPE * W;
		W_TYPE * dW;
		B_TYPE * B;
		B_TYPE * dB;

		W_TYPE * Sum;
		LAYER_IN_TYPE * in;
		LAYER_IN_TYPE * delta_in;

		int N_in;
		int N_out;
		//initial W and B size in the consturctor of sub class

		int W_size;
		int B_size;
		//accelerator memory
		//forward
		cl_mem W_acc;
		cl_mem B_acc;
		cl_mem in_acc;	
		cl_mem out_acc;
		//backward
		cl_mem dW_acc;
		cl_mem dB_acc;
		cl_mem delta_in_acc;
		cl_mem delta_out_acc;

		accelerator_cl * accelerator;

		string forward_program_name;
		cl_program forward_program;

		string forward_kernel_name;
		cl_kernel forward_kernel;	

		string backward_dWB_program_name;
		cl_program backward_dWB_program;

		string backward_delta_program_name;
		cl_program backward_delta_program;

		string backward_dWB_kernel_name;
		cl_kernel backward_dWB_kernel;	

		string backward_delta_kernel_name;
		cl_kernel backward_delta_kernel;	
	public:
		//add accelerator
		bool acc_flag;//decide if a accelerator is available whether to use it
		void add_accelerator(accelerator_cl & acc);
		//use accelerator to acclerate backward and forward
		virtual void forward_acc()=0;
		virtual void backward_acc()=0;
		void prepare_acc();
		void prepare_acc_forward();
		void trans_WB_to_acc();
		void trans_in_to_acc();
		void trans_out_to_host();
		void trans_dWB_to_host();
		void trans_dWB_to_acc();
		void trans_delta_in_to_acc();
		void trans_delta_out_to_host();

		//profile time class
		double forward_time;
		double backward_time;	
		void clear_time_count();
		void profile_time();
		void scalar_time(double scalar);
		const string name;
		LAYER_OUT_TYPE * out;
		LAYER_OUT_TYPE * delta_out;

		//construct and destruct
		forward_layer(const char * layer_name,int input_size,int output_size);
		~forward_layer();

		void set_in(LAYER_IN_TYPE * in_p);
		void set_delta_in(LAYER_OUT_TYPE * in_p);
		LAYER_OUT_TYPE get_out(int out_index);

		//scan W and B function
		void adjust_WB();
		void scalar_dWB(W_TYPE scalar);
		void clean_dWB();
		void display_W(int number);
		void display_dW(int number);
		void display_out(int number);
		void display_delta_out(int number);
		void display_in(int number);

		//interface to neural network class
		virtual void alloc_parameter()=0;
		virtual void initial_parameter()=0;
		virtual void forward()=0;
		virtual void backward()=0;
};
#endif
