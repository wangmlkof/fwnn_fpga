#ifndef SGD_LEARN
#define SGD_LEARN
#include "neural_network.h"
typedef unsigned char IMAGE_TYPE;

class sgd_learn
{
		protected:
				LAYER_IN_TYPE * train_data_set;
				LAYER_IN_TYPE * valid_data_set;
				LAYER_IN_TYPE * test_data_set;
				
				const int image_width;
				const int image_height;
				int batch_size;	
				int batch_total_num;
				W_TYPE learn_rate;
				neural_network * nn;
				const int train_size;
				const int valid_size;
				const int test_size;
		public:
				IMAGE_TYPE * train_image_set;
				LABEL_TYPE * train_label_set;

				IMAGE_TYPE * valid_image_set;
				LABEL_TYPE * valid_label_set;

				IMAGE_TYPE * test_image_set;
				LABEL_TYPE * test_label_set;

				//construct and destruct	
				sgd_learn(neural_network & neuralnetwork,int train,int valid,int test,int width,int height);
				~sgd_learn();
				void alloc_data_space();
				void pre_handle_data();
				void train_batch(int batch_num);//batch num start from 0
				void train_batch_acc(int batch_num);//batch num start from 0
				void train_whole_set();
				void train_whole_set_acc();
				float get_test_error();
				float get_test_error_acc();
				float get_valid_error();
				float get_valid_error_acc();

				void train_nn();
				//test accelerator effect
				void test_acc();
};
#endif
