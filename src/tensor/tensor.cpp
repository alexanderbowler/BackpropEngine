#include "backprop/tensor.hpp"
#include "internal_tensor.hpp"
#include <iostream>
namespace backprop{

void hello_world(){
    std::cout<<"hello world\n";
    impl::secret_implementation();
}

void impl::secret_implementation(){
    std::cout<<"secret\n";
}


}