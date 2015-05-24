__kernel void logreg1_forward(const int input_size,
				__global float * W_acc,
				__global float * B_acc,
				__global float * in_acc,
				__global float * out_acc
				)
{
		int i=get_global_id(0);
		float sum=0;
		for(int j=0;j<input_size;j++)
		{
			sum+=W_acc[i*input_size+j]*in_acc[j];
		}
		sum+=B_acc[i];
		out_acc[i]=exp(sum);
}
