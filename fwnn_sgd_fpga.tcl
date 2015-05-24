# SDAccel command script
# Design = smithwaterman example

# Define a project name
create_project -name fwnn_sgd_fpga -dir . -force

# Define the target platform of the application
set_property platform vc690-admpcie7v3-1ddr-gen2 [current_project]

set_property -name host_cflags -value "-g -Wall -D FPGA_DEVICE"  -objects [current_project]
# Host source files
add_files "main.cpp"
add_files "accelerator_cl.cpp"
add_files "conv_layer.cpp"
add_files "forward_layer.cpp"
add_files "full_connect_layer.cpp"
add_files "hidden_layer.cpp"
add_files "log_reg_layer.cpp"
add_files "maxpool_layer.cpp"
add_files "mnistio.cpp"
add_files "myclutils.cpp"
add_files "neural_network.cpp"
add_files "sgd_learn.cpp"

# Header files
add_files "accelerator_cl.h"
set_property file_type "c header files" [get_files "accelerator_cl.h"]
add_files "conv_layer.h"
set_property file_type "c header files" [get_files "conv_layer.h"]
add_files "forward_layer.h"
set_property file_type "c header files" [get_files "forward_layer.h"]
add_files "full_connect_layer.h"
set_property file_type "c header files" [get_files "full_connect_layer.h"]
add_files "hidden_layer.h"
set_property file_type "c header files" [get_files "hidden_layer.h"]
add_files "log_reg_layer.h"
set_property file_type "c header files" [get_files "log_reg_layer.h"]
add_files "maxpool_layer.h"
set_property file_type "c header files" [get_files "maxpool_layer.h"]
add_files "mnistio.h"
set_property file_type "c header files" [get_files "mnistio.h"]
add_files "myclutils.h"
set_property file_type "c header files" [get_files "myclutils.h"]
add_files "neural_network.h"
set_property file_type "c header files" [get_files "neural_network.h"]
add_files "sgd_learn.h"
set_property file_type "c header files" [get_files "sgd_learn.h"]

# Kernel definition
create_kernel hidden1_forward -type clc
add_files -kernel [get_kernels hidden1_forward] "./kernel/hidden1_forward.cl"

create_kernel hidden1_backward_dWB -type clc
add_files -kernel [get_kernels hidden1_backward_dWB] "./kernel/hidden1_backward_dWB.cl"

create_kernel hidden1_backward_delta -type clc
add_files -kernel [get_kernels hidden1_backward_delta] "./kernel/hidden1_backward_delta.cl"

create_kernel conv1_forward -type clc
add_files -kernel [get_kernels conv1_forward] "./kernel/conv1_forward.cl"

create_kernel conv1_backward_dWB -type clc
add_files -kernel [get_kernels conv1_backward_dWB] "./kernel/conv1_backward_dWB.cl"

create_kernel conv1_backward_delta -type clc
add_files -kernel [get_kernels conv1_backward_delta] "./kernel/conv1_backward_delta.cl"

create_kernel conv2_forward -type clc
add_files -kernel [get_kernels conv2_forward] "./kernel/conv2_forward.cl"

create_kernel conv2_backward_dWB -type clc
add_files -kernel [get_kernels conv2_backward_dWB] "./kernel/conv2_backward_dWB.cl"

create_kernel conv2_backward_delta -type clc
add_files -kernel [get_kernels conv2_backward_delta] "./kernel/conv2_backward_delta.cl"

# Define binary containers
create_opencl_binary -device [lindex [get_device "fpga0"] 0] fwnn_acc
set_property region "OCL_REGION_0" [get_opencl_binary fwnn_acc]

#create kernel
create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels hidden1_forward] -name hidden1_forward
create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels hidden1_backward_dWB] -name hidden1_backward_dWB
create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels hidden1_backward_delta] -name hidden1_backward_delta

create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels conv1_forward] -name conv1_forward
create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels conv1_backward_delta] -name conv1_backward_delta
create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels conv1_backward_dWB] -name conv1_backward_dWB

create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels conv2_forward] -name conv2_forward
create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels conv2_backward_delta] -name conv2_backward_delta
create_compute_unit -opencl_binary [get_opencl_binary fwnn_acc] -kernel [get_kernels conv2_backward_dWB] -name conv2_backward_dWB

# Run the design in CPU emulation mode
#compile_emulation -flow cpu -opencl_binary [get_opencl_binary fwnn_acc]
#run_emulation -flow cpu -args "fwnn_acc.xclbin"

# Run the design in hardware emulation mode
#compile_emulation -flow hardware -opencl_binary [get_opencl_binary fwnn_acc]
#run_emulation -flow hardware -args "fwnn_acc.xclbin"

# Generate the system estimate report
report_estimate

# Build the application for hardware
build_system

# Package the results for the card
package_system
