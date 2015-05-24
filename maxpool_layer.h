#ifndef MAXPOOL_LAYER_HEADER
#define MAXPOOL_LAYER_HEADER
#include "forward_layer.h"
class maxpool_layer:public forward_layer
{
		protected:
				const int pool_size;
				int pool_side_len;
		public:
				void alloc_parameter();
				void initial_parameter();

				void forward();
				void backward();

				void set_kernel_arg_forward();
				void forward_acc();
				void backward_acc();

				maxpool_layer(const char * layer_name,int input_size, int output_size);

};
#endif
