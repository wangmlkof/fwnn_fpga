#include "sgd_learn.h"
#include <math.h>
#include <time.h>
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
using namespace std;
extern void test_point();

sgd_learn::sgd_learn(neural_network & neuralnetwork,int train,int valid,int test,int width,int height):train_size(train),valid_size(valid),test_size(test),image_width(width),image_height(height)
{

		train_image_set=NULL;
		valid_image_set=NULL;
		test_image_set=NULL;

		train_label_set=NULL;
		valid_label_set=NULL;
		test_label_set=NULL;

		train_data_set=NULL;
		test_data_set=NULL;
		valid_data_set=NULL;
		nn=&neuralnetwork;

		batch_size=500;	
		learn_rate=0.1;
		batch_total_num=train_size/batch_size;
		//alloc data space
		alloc_data_space();
}

sgd_learn::~sgd_learn()
{
		if(train_image_set!=NULL)
		{
				delete []train_image_set;
		}
		if(valid_image_set!=NULL)
		{
				delete []valid_image_set;
		}
		if(test_image_set!=NULL)
		{
				delete []test_image_set;
		}
	
		if(train_label_set!=NULL)
		{
				delete []train_label_set;
		}
		if(valid_label_set!=NULL)
		{
				delete []valid_label_set;
		}
		if(test_label_set!=NULL)
		{
				delete []test_label_set;
		}

		if(train_data_set!=NULL)
		{
				delete []train_data_set;
		}
		if(test_data_set!=NULL)
		{
				delete []test_data_set;
		}
		if(valid_data_set!=NULL)
		{
				delete []valid_data_set;
		}
}

void sgd_learn::alloc_data_space()
{
		train_image_set=new IMAGE_TYPE[train_size*image_height*image_width];
		train_label_set=new LABEL_TYPE[train_size];
		cout<<"alloc train set array"<<endl;

		valid_image_set=new IMAGE_TYPE[valid_size*image_height*image_width];
		valid_label_set=new LABEL_TYPE[valid_size];
		cout<<"alloc valid set array"<<endl;

		test_image_set=new IMAGE_TYPE[test_size*image_height*image_width];
		test_label_set=new LABEL_TYPE[test_size];
		cout<<"alloc test set array"<<endl;
}

void sgd_learn::pre_handle_data()
{
		cout<<"pre handle data to float"<<endl;
		train_data_set=new LAYER_IN_TYPE[train_size*image_height*image_width];
		test_data_set=new LAYER_IN_TYPE[test_size*image_height*image_width];
		valid_data_set=new LAYER_IN_TYPE[valid_size*image_height*image_width];
		for(int i=0;i<train_size;i++)
		{
				for(int j=0;j<image_width*image_height;j++)
				{
						train_data_set[i*image_height*image_width+j]=train_image_set[i*image_width*image_height+j]/255.0;	
				}
		}

		for(int i=0;i<test_size;i++)
		{
				for(int j=0;j<image_width*image_height;j++)
				{
						test_data_set[i*image_height*image_width+j]=test_image_set[i*image_width*image_height+j]/255.0;	
				}
		}

		for(int i=0;i<valid_size;i++)
		{
				for(int j=0;j<image_width*image_height;j++)
				{
						valid_data_set[i*image_height*image_width+j]=valid_image_set[i*image_width*image_height+j]/255.0;	
				}
		}
		
		if(train_image_set!=NULL)
		{
				delete []train_image_set;
				train_image_set=NULL;
		}
		if(valid_image_set!=NULL)
		{
				delete []valid_image_set;
				valid_image_set=NULL;
		}
		if(test_image_set!=NULL)
		{
				delete []test_image_set;
				test_image_set=NULL;
		}
}

float sgd_learn::get_test_error()
{
		int error_num=0;
		//cout<<"test size:"<<test_size<<endl;
		for(int i=0;i<test_size;i++)
		{
				nn->set_sample(&(test_data_set[i*image_width*image_height]),test_label_set[i]);
				if(!nn->predict_sample())
				{
						error_num++;
				}
		}
		return error_num/(float)test_size;
}

float sgd_learn::get_test_error_acc()
{
		int error_num=0;
		//cout<<"test size:"<<test_size<<endl;
		nn->trans_WB_to_acc();
		for(int i=0;i<test_size;i++)
		{
				nn->set_sample(&(test_data_set[i*image_width*image_height]),test_label_set[i]);
				if(!nn->predict_sample_acc())
				{
						error_num++;
				}
		}
		return error_num/(float)test_size;
}

float sgd_learn::get_valid_error()
{
		int error_num=0;
		//	cout<<"valid size:"<<valid_size<<endl;
		for(int i=0;i<valid_size;i++)
		{
				nn->set_sample(&(valid_data_set[i*image_width*image_height]),valid_label_set[i]);
				if(!nn->predict_sample())
				{
						error_num++;
				}
		}
		return error_num/(float)valid_size;
}

float sgd_learn::get_valid_error_acc()
{
		int error_num=0;
		//	cout<<"valid size:"<<valid_size<<endl;
		nn->trans_WB_to_acc();
		for(int i=0;i<valid_size;i++)
		{
				nn->set_sample(&(valid_data_set[i*image_width*image_height]),valid_label_set[i]);
				if(!nn->predict_sample_acc())
				{
						error_num++;
				}
		}
		return error_num/(float)valid_size;
}

void sgd_learn::train_batch_acc(int batch_num)//batch num start from 0
{
		LAYER_IN_TYPE * batch_data_set=&(train_data_set[batch_num*batch_size*image_width*image_height]);
		LABEL_TYPE * batch_label_set=&(train_label_set[batch_num*batch_size]);
		//trans WB to acc
		nn->trans_WB_to_acc();

		//trans dWB to acc
		nn->clean_dWB();
		nn->trans_dWB_to_acc();

		nn->clear_time_count();
		//calculate delta W and B using bp
		for(int d=0;d<batch_size;d++)
		{
				LAYER_IN_TYPE * data=&(batch_data_set[d*image_width*image_height]);
				nn->set_sample(data,batch_label_set[d]);
				nn->train_sample_acc();
		}
		nn->scalar_time(1.0/batch_size);
		nn->profile_time();

		//adjust W and B
		//get dWB from accelerator memory
		nn->trans_dWB_to_host();//there is a bug!!! waiting to fix
		nn->scalar_dWB(learn_rate/(W_TYPE)batch_size);
		nn->adjust_WB();
}

void sgd_learn::train_batch(int batch_num)//batch num start from 0
{
		LAYER_IN_TYPE * batch_data_set=&(train_data_set[batch_num*batch_size*image_width*image_height]);
		LABEL_TYPE * batch_label_set=&(train_label_set[batch_num*batch_size]);

		nn->clean_dWB();

		nn->clear_time_count();
		//calculate delta W and B using bp
		for(int d=0;d<batch_size;d++)
		{
				LAYER_IN_TYPE * data=&(batch_data_set[d*image_width*image_height]);
				nn->set_sample(data,batch_label_set[d]);
				nn->train_sample();
		}
		//adjust W and B

		nn->scalar_time(1.0/batch_size);
		nn->profile_time();

		nn->scalar_dWB(learn_rate/(W_TYPE)batch_size);
		nn->adjust_WB();
}
void sgd_learn::train_whole_set()
{
	for(int i=0;i<batch_total_num;i++)
	{
		int start=clock();
		train_batch(i);
		int end=clock();
		cout<<"train on batch "<<i<<" using time "<<1.0*(end-start)/CLOCKS_PER_SEC<<" s"<<endl;
		if(i%10==9)
		{
			cout<<"test error : "<<get_test_error()<<endl;
		}
		//cout<<".";
		//fflush(stdout);
	}		
}
void sgd_learn::train_whole_set_acc()
{
	for(int i=0;i<batch_total_num;i++)
	{
		int start=clock();
		train_batch_acc(i);
		int end=clock();
		cout<<"train batch "<<i<<" using time "<<1.0*(end-start)/CLOCKS_PER_SEC<<" s"<<endl;
		if(i%10==9)
		{
			cout<<"test error : "<<get_test_error_acc()<<endl;
		}
		/*
		   cout<<".";
		   fflush(stdout);
		*/
		}		
}

void sgd_learn::train_nn()
{
		pre_handle_data();
//		cout<<"test error : "<<get_test_error()<<endl;
		for(int i=0;i<200;i++)
		{
				cout<<"train epoch : "<<i<<endl;
				int start=clock();
				train_whole_set_acc();
				//train_whole_set();
				int end=clock();
				cout<<"train on whole set using time "<<1.0*(end-start)/CLOCKS_PER_SEC<<" s"<<endl;
				cout<<"test error : "<<get_test_error_acc();
				//cout<<"test error : "<<get_test_error();
				cout<<endl;
		}
}

void sgd_learn::test_acc()
{
		pre_handle_data();
		train_batch_acc(0);
		//train_batch_acc(1);
		//train_whole_set();
}
