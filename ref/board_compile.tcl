# SDAccel command script
# Design = smithwaterman example

# Define a project name
create_project -name board_compilation_project -dir . -force

# Define the target platform of the application
set_property platform vc690-admpcie7v3-1ddr-gen2 [current_project]

# Host source files
add_files "main.cpp"
add_files "oclErrorCodes.cpp"
add_files "oclHelper.cpp"
add_files "soft.cpp"

# Header files
add_files "oclHelper.h"
set_property file_type "c header files" [get_files "oclHelper.h"]

# Kernel definition
create_kernel smithwaterman -type clc
add_files -kernel [get_kernels smithwaterman] "kernel_pipelined.cl"

# Define binary containers
create_opencl_binary -device [lindex [get_device "fpga0"] 0] test
set_property region "OCL_REGION_0" [get_opencl_binary test]
create_compute_unit -opencl_binary [get_opencl_binary test] -kernel [get_kernels smithwaterman] -name K1

# Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary test]

# Generate the system estimate report
report_estimate

# Run the design in CPU emulation mode
run_emulation -flow cpu -args "-d acc -k test.xclbin"

# Build the application for hardware
build_system

# Package the results for the card
package_system
