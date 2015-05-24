__kernel void conv2_backward_dWB(
				__global float * dW_acc, 
				__global float * dB_acc,
				__global float * in_acc,
				__global float * out_acc,
				__global float * delta_in_acc)
{
		int map_in_num=20;
		int map_in_side_len=12;
		int map_out_num=50;
		int map_out_side_len=8;
		int kernel_side_len=5;

		int l=get_global_id(0);
		int m=get_global_id(1);
		int n=get_global_id(2);
		int out_index=l*map_out_side_len*map_out_side_len+m*map_out_side_len+n;
		delta_in_acc[out_index]*=out_acc[out_index]*(1-out_acc[out_index]);
				for(int i=0;i<20;i++)
				{
					for(int j=0;j<5;j++)
					{
						for(int k=0;k<5;k++)
						{
							dW_acc[l*map_in_num*kernel_side_len*kernel_side_len+i*kernel_side_len*kernel_side_len+j*kernel_side_len+k]+=in_acc[i*map_in_side_len*map_in_side_len+(m+j)*map_in_side_len+(n+k)]*(-delta_in_acc[out_index]);
						}
					}
				}
				dB_acc[out_index]+=(-delta_in_acc[out_index]);

}
