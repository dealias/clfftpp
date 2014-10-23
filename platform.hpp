#include <vector>

#include <CL/cl.h>

//cl_device_id
void show_devices();
void create_device_tree(std::vector<std::vector<cl_device_id> > &D);
void find_platform_ids(std::vector<cl_platform_id > &A);
