#ifndef HIDDEN_LAYER
#define HIDDEN_LAYER

#include "full_connect_layer.h"
class hidden_layer :public full_connect_layer
{
	public:
		void initial_parameter();
		void forward();
		void backward();
		hidden_layer(const char * layer_name,int input_size,int output_size);

		//acc code
		void set_kernel_arg_forward();
		void set_kernel_arg_backward_dWB();
		void set_kernel_arg_backward_delta();
		void forward_acc();
		void backward_acc();
};
#endif
