#pragma once
#include <vector>
#include <unordered_map>

namespace backprop{

template <typename T>
class Tensor;

template<typename T>
class ConstantRegistry {
private:
    static std::unordered_map<T, std::unique_ptr<backprop::Tensor<T>>> constants_;
    
public:
    static backprop::Tensor<T>* get_constant(T value) {
        auto it = constants_.find(value);
        if (it != constants_.end()) {
            return it->second.get();
        }
        
        auto tensor = std::make_unique<backprop::Tensor<T>>(value);
        backprop::Tensor<T>* ptr = tensor.get();
        constants_[value] = std::move(tensor);
        return ptr;
    }
};

template<typename T>
std::unordered_map<T, std::unique_ptr<backprop::Tensor<T>>> ConstantRegistry<T>::constants_;

}