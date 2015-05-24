__kernel void hidden1_backward_dWB(
				__global float * dW_acc, 
				__global float * dB_acc,
				__global float * in_acc,
				__global float * out_acc,
				__global float * delta_in_acc)
{
		int input_size=800;
		int k=get_global_id(0);
		delta_in_acc[k]*=out_acc[k]*(1-out_acc[k]);
		for(int j=0;j<800;j++)
		{
			dW_acc[k*input_size+j]+=in_acc[j]*(-delta_in_acc[k]);
		}
		dB_acc[k]+=(-delta_in_acc[k]);
}
