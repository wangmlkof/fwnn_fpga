#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK
typedef unsigned char LABEL_TYPE;
#include "forward_layer.h"
#include "accelerator_cl.h"

class neural_network
{
		protected:
				int layer_num;
				int last_layer;
				int N_out;
				int N_in;
				W_TYPE * delta_in0;
				LABEL_TYPE correct_label;
				accelerator_cl * accelerator;
		public:
				//add accelerator
				void add_accelerator(accelerator_cl & acc);

				forward_layer ** layers;
				neural_network(int input_size,int output_size);
				~neural_network();
				
				//interface to sgd_learning class
				void set_sample(LAYER_IN_TYPE * sample_in,LABEL_TYPE answer);
				void train_sample();
				void train_sample_acc();
				bool predict_sample();
				bool predict_sample_acc();

				//assistive function
				void scalar_time(double scalar);
				void profile_time();
				void clear_time_count();
				void adjust_WB();
				void scalar_dWB(W_TYPE scalar);
				void clean_dWB();
				void forward();
				void backward();

				void forward_acc();
				void backward_acc();
				void trans_WB_to_acc();
				void trans_dWB_to_acc();
				void trans_dWB_to_host();
};
#endif
