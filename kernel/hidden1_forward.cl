__kernel void hidden1_forward(__global float * W_acc,
		     __global float * B_acc,
		     __global float * in_acc,
		     __global float * out_acc)
{
	int i=get_global_id(0);
	float sum=0;
	for(int j=0;j<800;j++)
	{
		sum+=W_acc[i*800+j]*in_acc[j];
	}
        sum+=B_acc[i];
        out_acc[i]=1/(1+exp(-sum));
}

