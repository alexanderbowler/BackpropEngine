#pragma once
#include <vector>
#include <type_traits>
#include <typeinfo>

#include "function.hpp"


namespace backprop{

template<typename T>
class Tensor{
    template <typename> friend class TensorTest;
    public:
        Tensor(T value): data_(value), shape_({}) {
            grad_fn_ptr = nullptr;
        }
        Tensor(T value, std::shared_ptr<Function<T>> grad_fn): data_(value), shape_({}), grad_fn_ptr(grad_fn) {}

        const T item() const{
            return data_;
        }

        const std::vector<int>& shape() const{
            return shape_;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){

            std::string output = "Tensor<" + std::string(typeid(T).name()) + ">(";
            os<<output;
            std::string shape = "";
            for(int dimension: tensor.shape()){
                shape += std::to_string(dimension) + ", ";
            }
            shape[shape.length()-2] = ')';
            os<<shape;
            std::string val = "{" + std::to_string(tensor.item()) + "}\n";
            os<<val;
            return os;
        }

        
        #ifdef UNIT_TEST
        const T get_data() const{
            return this->data_;
        }
        #endif

        // Calls the corresponding backward function
        void backward(){
            this->grad_fn_ptr->backward(*this);
        }

        

        std::shared_ptr<Function<T>> grad_fn_ptr;
        T grad_;
        friend class TensorTestAccess;  // Add this line
    protected:
        T data_;
        std::vector<int> shape_;


};

template<typename T, typename U>
Tensor<T> operator+(Tensor<T>& lfs, Tensor<U>& rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot add tensors of two different data types");

    
    return Tensor<T>(lfs.item() + rhs.item(), std::make_shared<AddFunction<T>>(&lfs, &rhs));
}

template<typename T, typename U>
Tensor<T> operator*(Tensor<T>& lfs, Tensor<U>& rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot multiply tensors of two different data types");
    
    return Tensor<T>(lfs.item() * rhs.item(), std::make_shared<MultiplyFunction<T>>(&lfs, &rhs));
}

}