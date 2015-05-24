__kernel void conv1_backward_delta(
				__global float * W_acc, 
				__global float * in_acc,
				__global float * delta_in_acc,
				__global float * delta_out_acc)
{
		int map_in_num=1;
		int map_in_side_len=28;
		int map_out_num=20;
		int map_out_side_len=24;
		int kernel_side_len=5;

		int i=get_global_id(0);
		int m=get_global_id(1);
		int n=get_global_id(2);

		for(int j=0;j<5;j++)
		{
			for(int k=0;k<5;k++)
			{
				int in_index=i*map_in_side_len*map_in_side_len+(m+j)*map_in_side_len+(n+k);
				delta_out_acc[in_index]=0;
				for(int l=0;l<20;l++)
				{
					int out_index=l*map_out_side_len*map_out_side_len+m*map_out_side_len+n;
					delta_out_acc[in_index]+=delta_in_acc[out_index]*W_acc[l*map_in_num*kernel_side_len*kernel_side_len+i*kernel_side_len*kernel_side_len+j*kernel_side_len+k];
				}
			}
		}
}
