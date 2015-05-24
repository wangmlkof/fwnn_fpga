#ifndef MNIST_HEADER
#define MNIST_HEADER

#include<stdio.h>

//used to change an byte array to an integer
int ba2int(unsigned char * buf,int byte_num);
typedef unsigned char IMAGE_TYPE;
typedef unsigned char LABEL_TYPE;
//used to store the file header of MNIST IMAGE
typedef struct 
{
		unsigned char magic_buf[4];
		unsigned char item_buf[4];
		unsigned char col_buf[4];
		unsigned char row_buf[4];
		int magic_num;
		int item_num;
		int col_num;
		int row_num;
} IMAGE_DB_HEADER;

//used to store the file header of MNIST LABEL 
typedef struct 
{
		unsigned char magic_buf[4];
		unsigned char item_buf[4];
		int magic_num;
		int item_num;
} LABEL_DB_HEADER;

//MNIST DATABASE CLASS
class mnist_db
{
	private:
		FILE * imagefp;
		FILE * labelfp;
		int count;
	public:
		IMAGE_DB_HEADER imagedb;
		LABEL_DB_HEADER labeldb;
		IMAGE_TYPE imagebuf[784];;
		LABEL_TYPE labelbuf;

		mnist_db(const char * image_file_name,const char * label_file_name);
		~mnist_db();
		int get_sample();
		int get_set(IMAGE_TYPE * image_set,LABEL_TYPE * label_set, int num);
};

#endif
