__kernel void maxpool1_forward(const int pool_side_len,
							   const int pool_size,
				__global float * in_acc,
				__global float * out_acc
				)
{
		int i=get_global_id(0);
		float max_value=in_acc[i*pool_size];
		for(int j=0;j<pool_side_len;j++)
		{
			for(int k=0;k<pool_side_len;k++)		
			{
				if(in_acc[i*pool_size+j*pool_side_len+k]>max_value)
				{
					max_value=in_acc[i*pool_size+j*pool_side_len+k];
				}
			}
		}
		out_acc[i]=max_value;
}
