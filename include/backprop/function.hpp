#pragma once
#include <vector>
/*
Jun 8 2025
Alex Bowler

Includes the information for which functions between tensors we support

*/
namespace backprop{
template <typename T>
class Tensor;
}
namespace backprop{

template <typename T>
class Function{
    public:
        std::vector<Tensor<T>*> parents;
        virtual void backward(Tensor<T>& output) = 0;  
};

template <typename T>
class AddFunction: public Function<T>{
    public:
    AddFunction(Tensor<T>* a, Tensor<T>* b){
        this->parents = {a, b};
    }
    void backward(Tensor<T>& output) override {
        this->parents[0]->grad_ = output.grad_;
        this->parents[1]->grad_ = output.grad_;
    }
};

template <typename T>
class MultiplyFunction : public Function<T>{
    public:
    MultiplyFunction(Tensor<T>* a, Tensor<T>* b){
        this->parents = {a, b};
    }
    void backward(Tensor<T>& output) override {
        this->parents[0]->grad_ = output.grad_ * this->parents[1]->item();
        this->parents[1]->grad_ = output.grad_ * this->parents[0]->item();
    }
};

}