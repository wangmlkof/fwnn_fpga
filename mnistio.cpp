#include "mnistio.h"
#include <iostream>
//convert byte arry to int
using namespace std;

int ba2int(unsigned char * buf,int byte_num)
{
		int result=0;
		int acc=1;
		for(int i=0;i<byte_num;i++)
		{
			result=result+acc*buf[(byte_num-1)-i];
			acc=acc*256;
		}
		return result;
}

int mnist_db::get_sample()
{
		if(count+1>imagedb.item_num)
		{
				return -1;
		}
		count++;
		fread(imagebuf,sizeof(unsigned char),imagedb.col_num*imagedb.row_num,imagefp);
		fread(&labelbuf,sizeof(unsigned char),1,labelfp);
		return 0;
}

int mnist_db::get_set(IMAGE_TYPE * image_set,LABEL_TYPE * label_set, int num)
{
		if((count+num)>imagedb.item_num)
		{
				cout<<"read out of database"<<endl;
				cout<<"remain:"<<imagedb.item_num-count<<"read:"<<num<<endl;
				return -1;
		}
		count+=num;
		fread(image_set,sizeof(unsigned char),imagedb.col_num*imagedb.row_num*num,imagefp);
		fread(label_set,sizeof(unsigned char),num,labelfp);
		return 0;
}

mnist_db::mnist_db(const char * image_file_name,const char * label_file_name)
{
		count=0;
		imagefp=NULL;
		labelfp=NULL;

		imagefp=fopen(image_file_name,"rb");
		fread(&imagedb,sizeof(unsigned char),16,imagefp);
		imagedb.magic_num=ba2int(imagedb.magic_buf,4);
		imagedb.item_num=ba2int(imagedb.item_buf,4);
		imagedb.col_num=ba2int(imagedb.col_buf,4);
		imagedb.row_num=ba2int(imagedb.row_buf,4);

		labelfp=fopen(label_file_name,"rb");
		fread(&labeldb,sizeof(unsigned char),8,labelfp);
		labeldb.magic_num=ba2int(labeldb.magic_buf,4);
		labeldb.item_num=ba2int(labeldb.item_buf,4);
}

mnist_db::~mnist_db()
{
		if(imagefp!=NULL)
		{
				fclose(imagefp);
		}
		if(labelfp!=NULL)
		{
				fclose(labelfp);
		}
}
