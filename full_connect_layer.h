#ifndef FULL_CONNECT_LAYER
#define FULL_CONNECT_LAYER
#include "forward_layer.h"

class full_connect_layer:public forward_layer
{
		public:
				void compute_sum();
				void alloc_parameter();
				full_connect_layer(const char * layer_name,int input_size, int output_size);
};

#endif
