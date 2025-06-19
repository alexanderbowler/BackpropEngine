#pragma once
#include <vector>
/*
Jun 8 2025
Alex Bowler

Includes the function classes which represent which functions we can call between tensors

*/

namespace backprop{

template <typename T>
class Tensor;

/**
 * @brief Base Function class to be inherited by specific operation functions.
 * 
 * This class represents a differentiable operation in the computation graph.
 * Each Function instance keeps track of its parent tensors and implements
 * a backward method to propagate gradients during backpropagation.
 * 
 * @tparam T The data type of the tensor elements (e.g., float, double).
 */
template <typename T>
class Function{
    public:
        std::vector<Tensor<T>*> parents;
        virtual void backward(Tensor<T>& output) = 0;  
};

/**
 * @brief Function representing element-wise addition of two tensors.
 * 
 * The AddFunction class implements the addition operation in the computation graph.
 * It stores pointers to the two parent tensors being added. During backpropagation,
 * the gradient from the output is propagated equally to both parent tensors, since
 * d/dx (x + y) = 1 for both x and y.
 * 
 * @tparam T The data type of the tensor elements (e.g., float, double).
 */
template <typename T>
class AddFunction: public Function<T>{
    public:
    /**
     * @brief Constructs an AddFunction with two parent tensors.
     * 
     * @param a Pointer to the first parent tensor.
     * @param b Pointer to the second parent tensor.
     */
    AddFunction(Tensor<T>* a, Tensor<T>* b){
        this->parents = {a, b};
    }

    /**
     * @brief Backward pass for the addition operation.
     * 
     * Adds the output gradient to both parent tensors' gradients.
     * 
     * @param output The output tensor from which the gradient is propagated.
     */
    void backward(Tensor<T>& output) override {
        this->parents[0]->grad_ += output.grad_;
        this->parents[1]->grad_ += output.grad_;
    }
};
/**
 * @brief Function representing element-wise multiplication of two tensors.
 * 
 * The MultiplyFunction class implements the multiplication operation in the computation graph.
 * It stores pointers to the two parent tensors being multiplied. During backpropagation,
 * the gradient from the output is propagated to both parent tensors, using the product rule:
 * d/dx (x * y) = y, d/dy (x * y) = x.
 * 
 * @tparam T The data type of the tensor elements (e.g., float, double).
 */
template <typename T>
class MultiplyFunction : public Function<T>{
    public:
    /**
     * @brief Constructs a MultiplyFunction with two parent tensors.
     * 
     * @param a Pointer to the first parent tensor.
     * @param b Pointer to the second parent tensor.
     */
    MultiplyFunction(Tensor<T>* a, Tensor<T>* b){
        this->parents = {a, b};
    }

    /**
     * @brief Backward pass for the multiplication operation.
     * 
     * Propagates the output gradient to both parent tensors using the product rule.
     * 
     * @param output The output tensor from which the gradient is propagated.
     */
    void backward(Tensor<T>& output) override {
        this->parents[0]->grad_ += output.grad_ * this->parents[1]->item();
        this->parents[1]->grad_ += output.grad_ * this->parents[0]->item();
    }
};

/**
 * @brief Function representing the hyperbolic tangent (tanh) operation on a tensor.
 * 
 * The TanhFunction class implements the tanh activation function in the computation graph.
 * It stores a pointer to the parent tensor. During backpropagation, the gradient from the output
 * is propagated to the parent tensor using the derivative of tanh:
 * d/dx tanh(x) = 1 - tanh(x)^2.
 * 
 * @tparam T The data type of the tensor elements (e.g., float, double).
 */
template <typename T>
class TanhFunction : public Function<T>{
    public:
    /**
     * @brief Constructs a TanhFunction with a parent tensor.
     * 
     * @param parent Pointer to the parent tensor.
     */
    TanhFunction(Tensor<T>* parent){
        this->parents = {parent};
    }

    /**
     * @brief Backward pass for the tanh operation.
     * 
     * Propagates the output gradient to the parent tensor using the derivative of tanh.
     * 
     * @param output The output tensor from which the gradient is propagated.
     */
    void backward(Tensor<T>& output) override {
        T tanh_x = output.item();
        this->parents[0]->grad_ += output.grad_ * (1 - tanh_x * tanh_x);
    }
};

}