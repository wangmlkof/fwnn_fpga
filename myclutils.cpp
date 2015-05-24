#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include "myclutils.h"
using namespace std;
int load_file_to_memory(const char *filename, char **result)
{ 
	int size = 0;
	FILE *f = fopen(filename, "rb");
	if (f == NULL) 
	{ 
		*result = NULL;
		cout<<"fail to open xclbin file"<<endl;
		return -1; // -1 means file opening fail 
	} 
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size+1);
	if (size != fread(*result, sizeof(char), size, f)) 
	{ 
		free(*result);
		cout<<"fail to read xclbin file"<<endl;
		return -2; // -2 means file reading fail 
	} 
	fclose(f);
	(*result)[size] = 0;
	return size;
}
cl_context CreateGPUContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;
	// First, select an OpenCL platform to run on.
	// For this example, we simply choose the first available
	// platform. Normally, you would query for all available
	// platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Failed to find any OpenCL platforms");
		return NULL;
	}
	// Next, create an OpenCL context on the platform. Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	//create gpu context
	context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_GPU,NULL, NULL, &errNum);

	if (errNum != CL_SUCCESS)
	{
		printf( "Could not create GPU context, trying CPU...\n");

	}
	return context;
}
cl_context CreateCPUContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;
	// First, select an OpenCL platform to run on.
	// For this example, we simply choose the first available
	// platform. Normally, you would query for all available
	// platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Failed to find any OpenCL platforms");
		return NULL;
	}
	// Next, create an OpenCL context on the platform. Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	//create context of GPU
	context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_CPU,NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		printf("Failed to create an OpenCL GPU or CPU context.\n");;
	}

	return context;
}

cl_context CreateFPGAContext()
{
	cl_int err;
	cl_platform_id platform_id;         // platform id
	cl_device_id device_id;             // compute device id 
	cl_context context = NULL;

	char cl_platform_vendor[1001];
	char cl_platform_name[1001];

	err = clGetPlatformIDs(1,&platform_id,NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to find an OpenCL platform!\n");
		printf("Test failed\n");
		return NULL;
	}
	err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
		printf("Test failed\n");
		return NULL;
	}
	printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
	err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
		printf("Test failed\n");
		return NULL;
	}
	printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
	// Connect to a compute device
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR ,
			1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		printf("Test failed\n");
		return NULL;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		printf("Test failed\n");
		return NULL;
	}
	return context;
}


cl_command_queue CreateCommandQueue(cl_context context,cl_device_id *device,int device_num)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;
	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL,
			&deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		printf( "Failed call to	clGetContextInfo(...,GL_CONTEXT_DEVICES,...)");
		return NULL;
	}
	if (deviceBufferSize <= 0)
	{
		printf( "No devices available.");
		return NULL;
	}
	// Allocate memory for the devices buffer
	devices = new cl_device_id[deviceBufferSize/sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES,deviceBufferSize, devices, NULL);
	if(errNum != CL_SUCCESS)
	{
		printf( "Failed to get device IDs\n");
		return NULL;
	}
	// In this example, we just choose the first available device.
	// In a real program, you would likely use all available
	// devices or choose the highest performance device based on
	// OpenCL device queries.
	commandQueue = clCreateCommandQueue(context,
			devices[device_num], 0, NULL);
	if (commandQueue == NULL)
	{
		printf("Failed to create commandQueue for device %d\n",device_num);
		return NULL;
	}
	*device = devices[device_num];
	delete [] devices;
	return commandQueue;
}

cl_program CreateProgram_src(cl_context context, cl_device_id device,const char* fileName)
{
	cl_int errNum;
	cl_program program;
	ifstream kernelFile(fileName, ios::in);
	if (!kernelFile.is_open())
	{
		cerr << "Failed to open file for reading: " << fileName <<
			endl;
		return NULL;
	}
	ostringstream oss;
	oss << kernelFile.rdbuf();
	string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,(const char**)&srcStr,NULL, NULL);
	if (program == NULL)
	{
		cerr << "Failed to create CL program from source." << endl;
		return NULL;
	}
	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), buildLog, NULL);
		cerr << "Error in kernel: " << endl;
		cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}
	return program;
}
cl_program CreateProgram_bin(cl_context context, cl_device_id device,const char* fileName)
{
	int status;
	cl_int err;
	cl_program program;
	unsigned char *kernelbinary;
	printf("loading %s\n", fileName);
	int n_i = load_file_to_memory(fileName, (char **) &kernelbinary);
	if (n_i < 0) {
		printf("failed to load kernel from xclbin: %s\n", fileName);
		printf("Test failed\n");
		return NULL;
	}
	size_t n = n_i;
	// Create the compute program from offline
	program = clCreateProgramWithBinary(context, 1, &device, &n,
			(const unsigned char **) &kernelbinary, &status, &err);
	if ((!program) || (err!=CL_SUCCESS)) {
		printf("Error: Failed to create compute program from binary %d!\n", err);
		printf("Test failed\n");
		return NULL;
	}

	// Build the program executable
	//
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		printf("Test failed\n");
		return NULL;
	}
	return program;
}
const char *getErrorString(cl_int error)
{
	switch(error){
		// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

			  // compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

			  // extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
	}
}
void DisplayPlatformInfo(
		cl_platform_id id, 
		cl_platform_info name,
		string str)
{
	cl_int errNum;
	size_t paramValueSize;
	//cout <<id <<"!" <<endl;

	errNum = clGetPlatformInfo(
			id,
			name,
			0,
			NULL,
			& paramValueSize);
	if(errNum != CL_SUCCESS)
	{
		cerr << "Failed to find OpenCL platform" << str <<"." <<endl;
		return;	
	}

	char * info = (char*)alloca(sizeof(char) * paramValueSize);
	errNum = clGetPlatformInfo(
			id,
			name,
			paramValueSize,
			info,
			NULL);
	if(errNum != CL_SUCCESS)
	{
		cerr << "Failed to find OpenCL platform" << str <<"." <<endl;
		return;		
	}
	cout <<str <<":\t" <<info <<endl;
}

void DisplayDeviceInfo(
		cl_device_id id, 
		cl_device_info name,
		string str)
{
	cl_int errNum;
	size_t paramValueSize;
	//cout <<id <<"!" <<endl;

	errNum = clGetDeviceInfo(
			id,
			name,
			0,
			NULL,
			& paramValueSize);
	if(errNum != CL_SUCCESS)
	{
		cerr << "Failed to find OpenCL device" << str <<"." <<endl;
		system("pause");
		return;	
	}

	void * info = (void*)alloca(sizeof(char) * paramValueSize);
	//cl_int info;
	errNum = clGetDeviceInfo(
			id,
			name,
			paramValueSize,
			info,
			NULL);
	if(errNum != CL_SUCCESS)
	{
		cerr << "Failed to find OpenCL device" << str <<"." <<endl;
		return;		
	}
	cout <<str <<":\t" << info <<endl;
}



void GetDevices(cl_platform_id id)
{
	cl_int errNum;
	cl_uint numDevices;
	cl_device_id * deviceIds;

	//cout <<id  <<endl;

	errNum = clGetDeviceIDs(
			id,
			CL_DEVICE_TYPE_ALL,
			0,
			NULL,
			&numDevices);

	if(errNum != CL_SUCCESS || numDevices < 1)
	{
		cout <<"No device found for platform" << id <<endl;
		system("pause");
		exit(1);
	}

	deviceIds = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);

	errNum = clGetDeviceIDs(
			id,
			CL_DEVICE_TYPE_ALL,
			numDevices,
			deviceIds,
			NULL);

	if(errNum != CL_SUCCESS || numDevices < 1)
	{
		cout <<"No device found for platform" << id <<endl;
		system("pause");
		exit(1);
	}

	cout <<endl <<numDevices <<" devives found on the platform!" <<endl;

	char buffer[1024];
	cl_uint buf_uint;
	size_t buf_size_t[3];
	cl_bool support_image;


	for(cl_uint i = 0; i < numDevices; i ++)
	{
		cout <<i+1 <<":" <<endl;



		errNum = clGetDeviceInfo(deviceIds[i],CL_DEVICE_NAME,sizeof(buffer),buffer,NULL);
		if(errNum != CL_SUCCESS || numDevices < 1)
		{
			cout <<"get device info error!" << "CL_DEVICE_VENDOR_ID" <<endl;
			system("pause");
			exit(1);
		}
		cout <<"CL_DEVICE_NAME:" <<buffer <<endl;

		errNum = clGetDeviceInfo(deviceIds[i],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&buf_uint,NULL);
		if(errNum != CL_SUCCESS || numDevices < 1)
		{
			cout <<"get device info error!" << "CL_DEVICE_MAX_COMPUTE_UNITS" <<endl;
			system("pause");
			exit(1);
		}
		cout <<"CL_DEVICE_MAX_COMPUTE_UNITS:" <<buf_uint <<endl;

		errNum = clGetDeviceInfo(deviceIds[i],CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,sizeof(cl_uint),&buf_uint,NULL);
		if(errNum != CL_SUCCESS || numDevices < 1)
		{
			cout <<"get device info error!" << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS" <<endl;
			system("pause");
			exit(1);
		}
		cout <<"CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:" <<buf_uint <<endl;

		errNum = clGetDeviceInfo(deviceIds[i],CL_DEVICE_MAX_WORK_ITEM_SIZES,sizeof(buf_size_t),buf_size_t,NULL);
		if(errNum != CL_SUCCESS || numDevices < 1)
		{
			cout <<"get device info error!" << "CL_DEVICE_MAX_WORK_ITEM_SIZES" <<endl;
			system("pause");
			exit(1);
		}
		cout <<"CL_DEVICE_MAX_WORK_ITEM_SIZES:" <<"( " << buf_size_t[0] <<", " <<buf_size_t[1] <<", " <<buf_size_t[2] <<")" <<endl;
		DisplayDeviceInfo(deviceIds[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,"CL_DEVICE_MAX_WORK_ITEM_SIZES");

		errNum = clGetDeviceInfo(deviceIds[i],CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(cl_uint),&buf_uint,NULL);
		if(errNum != CL_SUCCESS || numDevices < 1)
		{
			cout <<"get device info error!" << "CL_DEVICE_MAX_CLOCK_FREQUENCY" <<endl;
			system("pause");
			exit(1);
		}
		cout <<"CL_DEVICE_MAX_CLOCK_FREQUENCY:" <<buf_uint << "MHz" <<endl;

		errNum = clGetDeviceInfo(deviceIds[i],CL_DEVICE_IMAGE_SUPPORT,sizeof(cl_bool),&support_image,NULL);
		if(errNum != CL_SUCCESS || numDevices < 1)
		{
			cout <<"get device info error!" << "CL_DEVICE_IMAGE_SUPPORT" <<endl;
			system("pause");
			exit(1);
		}
		cout <<"CL_DEVICE_IMAGE_SUPPORT:";
		if(support_image == CL_TRUE) cout<<"CL_TRUE" <<endl;
		else cout <<"CL_FALSE" <<endl;

		//cout <<CL_FALSE <<endl;
	}
	//cout << numDevices <<endl;

}
