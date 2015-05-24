__kernel void hidden1_backward_delta(
				__global float * W_acc, 
				__global float * in_acc,
				__global float * delta_in_acc,
				__global float * delta_out_acc)
{
		int input_size=800;
		int output_size=500;

		int k=get_global_id(0);
		delta_out_acc[k]=0;
		for(int m=0;m<500;m++)
		{
			delta_out_acc[k]+=delta_in_acc[m]*W_acc[m*input_size+k];
		}	
}
