#ifndef CONV_LAYER_HEADER
#define CONV_LAYER_HEADER
#include "forward_layer.h"
class conv_layer: public forward_layer
{
		protected:
				const int map_in_num;
				const int map_in_side_len;
				const int map_out_num;
				const int map_out_side_len;
				const int kernel_side_len;
		public:
				void alloc_parameter();
				void initial_parameter();

				void forward();
				void backward();

				void set_kernel_arg_backward_dWB();
				void set_kernel_arg_backward_delta();
				void set_kernel_arg_forward();
				void forward_acc();
				void backward_acc();

				conv_layer(const char * layer_name,
							int mapinnum,
							int mapinsidelen,
							int mapoutnum,
							int mapoutsidelen,
							int kernelsidelen);


};
#endif
