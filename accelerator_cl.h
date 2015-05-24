#ifndef ACCELERATOR_CL_HEADER
#define ACCELERATOR_CL_HEADER
#include <CL/opencl.h>
#include <string>
using namespace std;
class accelerator_cl
{
	protected:
		cl_context platform;
		cl_device_id device;
		cl_int err_info;
		cl_command_queue command;
		cl_program program;
		string kernel_path;
		string xclbin_name;
	public:
		friend class forward_layer;
		friend class conv_layer;
		friend class maxpool_layer;;
		friend class hidden_layer;
		friend class log_reg_layer;
		accelerator_cl();
		~accelerator_cl();
		//get platform infomation
		void platform_info();

};

#endif
