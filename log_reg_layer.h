#ifndef LOG_REG_LAYER
#define LOG_REG_LAYER
#include "full_connect_layer.h"

class log_reg_layer: public full_connect_layer
{
		public:
				void initial_parameter();
				void forward();
				void backward();

				void set_kernel_arg_forward();
				void forward_acc();
				void backward_acc();

				log_reg_layer(const char * layer_name,int input_size,int output_size);
};
#endif
