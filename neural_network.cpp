#include "neural_network.h"
#include "log_reg_layer.h"
#include "hidden_layer.h"
#include "maxpool_layer.h"
#include "conv_layer.h"
#include <time.h>
#include <iostream>
#include <stdio.h>
using namespace std;
extern void test_point();
neural_network::neural_network(int input_size,int output_size)
{
		cout<<"constuct neural network"<<endl;
		accelerator=NULL;

		layer_num=6;

		N_in=input_size;
		N_out=output_size;

		delta_in0=new W_TYPE[N_out];
		last_layer=layer_num-1;
		layers =new forward_layer *[layer_num];
		layers[5]=new log_reg_layer("logreg1", 500,N_out);
		layers[4]=new hidden_layer( "hidden1",800,500);
		layers[3]=new maxpool_layer("maxpool2",3200,800);
		layers[2]=new conv_layer("conv2",20,12,50,8,5);
		layers[1]=new maxpool_layer("maxpool1",11520,2880);
		layers[0]=new conv_layer("conv1",1,28,20,24,5);
		//construct network
		//backward path
		layers[last_layer]->set_delta_in(delta_in0);
		for(int i=0;i<last_layer;i++)
		{
				layers[i]->set_delta_in(layers[i+1]->delta_out);
		}
		//forward path
		for(int i=0;i<last_layer;i++)
		{
				layers[i+1]->set_in(layers[i]->out);
		}
}

neural_network::~neural_network()
{
		delete []layers;
		cout<<"destuct neural network"<<endl;
}

void neural_network::add_accelerator(accelerator_cl & acc)
{	
	accelerator=&acc;
	for(int i=0;i<layer_num;i++)
	{
		layers[i]->add_accelerator(acc);
	}
	accelerator->platform_info();
}

void neural_network::set_sample(LAYER_IN_TYPE * sample_in,LABEL_TYPE answer)
{
		layers[0]->set_in(sample_in);
		correct_label=answer;
}

bool neural_network::predict_sample()
{
	LABEL_TYPE max=0;	
	forward();
	for(int i=1;i<N_out;i++)
	{
			if(layers[last_layer]->get_out(i)>layers[last_layer]->get_out(max))
			{
					max=i;
			}
	}
	
	return correct_label==max;
}

bool neural_network::predict_sample_acc()
{
	LABEL_TYPE max=0;	
	forward_acc();
	for(int i=1;i<N_out;i++)
	{
			if(layers[last_layer]->get_out(i)>layers[last_layer]->get_out(max))
			{
					max=i;
			}
	}
	return correct_label==max;
}

void neural_network::train_sample()
{
		forward();
		for(int k=0;k<N_out;k++)
		{
				if(k==correct_label)
				{
						delta_in0[k]=-(1-layers[last_layer]->get_out(k));
				}	
				else
				{
						delta_in0[k]=layers[last_layer]->get_out(k);
				}
		}
		backward();
}

void neural_network::train_sample_acc()
{
		forward_acc();
		for(int k=0;k<N_out;k++)
		{
				if(k==correct_label)
				{
						delta_in0[k]=-(1-layers[last_layer]->get_out(k));
				}	
				else
				{
						delta_in0[k]=layers[last_layer]->get_out(k);
				}
		}
		backward_acc();
}

//assistive function
void neural_network::adjust_WB()
{
		for(int i=0;i<layer_num;i++)
		{
				layers[i]->adjust_WB();
		}
}

void neural_network::scalar_dWB(W_TYPE scalar)
{
		for(int i=0;i<layer_num;i++)
		{
				layers[i]->scalar_dWB(scalar);
		}
}

void neural_network::clean_dWB()
{
		for(int i=0;i<layer_num;i++)
		{
				layers[i]->clean_dWB();
		}
}

void neural_network::forward()
{
		for(int i=0;i<layer_num;i++)
		{
				int start=clock();
				layers[i]->forward();
				int end =clock();
				layers[i]->forward_time+=(1.0*(end-start)/CLOCKS_PER_SEC);
		}
}

void neural_network::backward()
{
		for(int i=last_layer;i>=0;i--)
		{
				int start=clock();
				layers[i]->backward();
				//layers[i]->display_delta_out(1);
				//test_point();
				int end =clock();
				layers[i]->backward_time+=(1.0*(end-start)/CLOCKS_PER_SEC);
		}
}
void neural_network::trans_WB_to_acc()
{
	for(int i=0;i<layer_num;i++)
	{
		if(layers[i]->acc_flag==1)
		{
			layers[i]->trans_WB_to_acc();
		}
	}
}

void neural_network::trans_dWB_to_acc()
{
	for(int i=0;i<layer_num;i++)
	{
		if(layers[i]->acc_flag==1)
		{
			layers[i]->trans_dWB_to_acc();
		}
	}
}

void neural_network::trans_dWB_to_host()
{
	for(int i=0;i<layer_num;i++)
	{
		if(layers[i]->acc_flag==1)
		{
			layers[i]->trans_dWB_to_host();
		}
	}
}

void neural_network::forward_acc()
{
		for(int i=0;i<layer_num;i++)
		{
				int start=clock();
				layers[i]->forward_acc();
				int end =clock();
				layers[i]->forward_time+=(1.0*(end-start)/CLOCKS_PER_SEC);
		}
}

void neural_network::backward_acc()
{
		for(int i=last_layer;i>=0;i--)
		{
				int start=clock();
				layers[i]->backward_acc();
				int end =clock();
				layers[i]->backward_time+=(1.0*(end-start)/CLOCKS_PER_SEC);
		}
}

void neural_network::clear_time_count()
{
		for(int i=0;i<layer_num;i++)
		{
				layers[i]->clear_time_count();
		}
}

void neural_network::profile_time()
{
		cout<<"layers"<<"   "<<"forward"<<"   "<<"backward"<<endl;
		for(int i=0;i<layer_num;i++)
		{
				layers[i]->profile_time();
		}
}
void neural_network::scalar_time(double scalar)
{
		for(int i=0;i<layer_num;i++)
		{
				layers[i]->scalar_time(scalar);
		}
}
