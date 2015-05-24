#include <stdio.h>
#include <time.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include "mnistio.h"
#include "sgd_learn.h"
#include "neural_network.h"

#define TRAIN_SIZE 50000
#define VALID_SIZE 10000
#define TEST_SIZE 10000

using namespace std;
/* ref code http://deeplearning.net/tutorial/logreg.html */
void test_point()
{
	char c= getchar();
	if(c=='q')
		exit(0);
}
double gaussrand()
{
		static double V1, V2, S;
		static int phase = 0;
		double X;

		if ( phase == 0 ) {
				do {
						double U1 = (double)rand() / RAND_MAX;
						double U2 = (double)rand() / RAND_MAX;

						V1 = 2 * U1 - 1;
						V2 = 2 * U2 - 1;
						S = V1 * V1 + V2 * V2;
				} while(S >= 1 || S == 0);

				X = V1 * sqrt(-2 * log(S) / S);
		} else
				X = V2 * sqrt(-2 * log(S) / S);

		phase = 1 - phase;

		return X;
}

#define TRAIN_SIZE 50000
#define TEST_SIZE 10000
#define VALID_SIZE 10000
#define IMAGE_HEIGHT 28
#define IMAGE_WIDTH 28
int main(int argc, char * argv[])
{
		//first read the mnist data
		cout<<"open mnist database"<<endl;
		mnist_db train_db("/home/wml/project/wml/mnist_db/train-images.idx3-ubyte",
				"/home/wml/project/wml/mnist_db/train-labels.idx1-ubyte");
		mnist_db test_db("/home/wml/project/wml/mnist_db/t10k-images.idx3-ubyte",
				"/home/wml/project/wml/mnist_db/t10k-labels.idx1-ubyte");

		neural_network nn(784,10);
		cout<<"add accelerator to neural network"<<endl;
		accelerator_cl fpga_acc;
		nn.add_accelerator(fpga_acc);
		//sgd learn method, minibatch
		sgd_learn sgd(nn,TRAIN_SIZE,VALID_SIZE,TEST_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT);
		//init train data set
		train_db.get_set(sgd.train_image_set,sgd.train_label_set,TRAIN_SIZE);
		//init valid data set
		train_db.get_set(sgd.valid_image_set,sgd.valid_label_set,VALID_SIZE);
		//init test data set
		test_db.get_set(sgd.test_image_set,sgd.test_label_set,TEST_SIZE);
		//init log regression neural network
		//cout<<"start trainning"<<endl;
		sgd.train_nn();
		//sgd.test_acc();
		return 0;
}
