#include "backprop/tensor.hpp"
#include <iostream>
#include <string>

int main(){
    backprop::Tensor<float> float_tensor(3.0);
    backprop::Tensor<float> second_tensor(4.0);
    std::cout<<"Tensor value "<<float_tensor.item()<<std::endl;
    std::cout<<float_tensor;
    // backprop::Tensor<float> sum = float_tensor + second_tensor;
    // std::cout<<sum<<std::endl;
    return 0;
}