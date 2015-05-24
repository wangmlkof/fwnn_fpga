#include "accelerator_cl.h"
#include "myclutils.h"
#include <iostream>
using namespace std;
accelerator_cl::accelerator_cl()
{
	cout<<"consturct accelerator"<<endl;
	//init platform context
	platform=CreateFPGAContext();
	//create command line queue
	command=CreateCommandQueue(platform,&device,0);
	//prefix of kernel code
	kernel_path.append("./kernel/");
	xclbin_name.append("fwnn_acc.xclbin");
	program=CreateProgram_bin(platform,device,xclbin_name.c_str());
}

accelerator_cl::~accelerator_cl()
{
	clReleaseCommandQueue(command);
	clReleaseContext(platform);
}

void accelerator_cl::platform_info()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id * platformIds;

	/*First, query the total unmber of platforms*/
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	if(errNum != CL_SUCCESS || numPlatforms <= 0)
	{   
		cerr << "Failed to find any OpenCL platform." << endl;
		return;
	}   

	/*Next, allocate memory for the installed platforms, and query to get the list.*/
	platformIds = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);

	errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);
	if(errNum != CL_SUCCESS || numPlatforms <= 0)
	{   
		cerr << "Failed to find any OpenCL platform." << endl;
		return;
	}   

	cout << "Number of platforms:\t" <<numPlatforms <<endl;

	for(cl_uint i = 0; i < numPlatforms; i++)
	{   
		DisplayPlatformInfo(platformIds[i], CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
		DisplayPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
		DisplayPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
		DisplayPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
		/*DisplayPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");*/
		GetDevices(platformIds[i]);
		cout<<endl;
	}
}

