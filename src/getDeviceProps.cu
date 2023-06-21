#include <iostream>
using namespace std;


int main()
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    // Reference to the complete structure ->  https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
    cout<<"Number of devices: "<<dev_count<<'\n';

    cudaDeviceProp dev_prop;
    for(int i = 0 ; i < dev_count; i++){
        cudaGetDeviceProperties(&dev_prop, i);
        cout<<"Props of device ["<< i+1  << "]:\n";
        cout<<" -> Number of SMs: "<<dev_prop.multiProcessorCount<<'\n';
        cout<<" -> Max Threads per Block: "<<dev_prop.maxThreadsPerBlock<<'\n';
        cout<<" -> Max block dim: "<<"("<<dev_prop.maxThreadsDim[0]<<", "<<dev_prop.maxThreadsDim[1]<<", "<<dev_prop.maxThreadsDim[2]<<")\n";
        cout<<" -> Max grid dim: "<<"("<<dev_prop.maxGridSize[0]<<", "<<dev_prop.maxGridSize[1]<<", "<<dev_prop.maxGridSize[2]<<")\n";
        cout<<" -> Clock Rate: "<<dev_prop.clockRate<<'\n';
    }
}