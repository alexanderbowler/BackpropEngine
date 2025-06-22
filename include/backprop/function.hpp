#pragma once
#include <vector>
#include <cassert>
#include <memory>
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
        // pointer to the parent tensors, Note does not pass ownership
        std::vector<Tensor<T>*> parents;
        // pointer to the tensor that the function created
        Tensor<T>* output_ = nullptr;
        virtual void backward() = 0;  
        virtual void forward() = 0;

        /**
         * @brief Sets the pointer to the tensor that this function created.
         * 
         * This method is used to associate the function instance with the output tensor
         * it produces in the computation graph. This association is useful for tracking
         * the relationship between operations and their resulting tensors, especially
         * during backpropagation.
         * 
         * @param o Pointer to the tensor created by this function.
         */
        void set_output_tensor(Tensor<T>* o){
            this->output_ = o;
        }
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
     */
    void backward() override {
        assert(this->output_ != nullptr);
        this->parents[0]->grad_ += this->output_->grad_;
        this->parents[1]->grad_ += this->output_->grad_;
    }

    void forward() override {
        assert(this->output_ != nullptr);
        this->output_->set(this->parents[0]->item() + this->parents[1]->item());
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
    void backward() override {
        assert(this->output_ != nullptr);
        this->parents[0]->grad_ += this->output_->grad_ * this->parents[1]->item();
        this->parents[1]->grad_ += this->output_->grad_ * this->parents[0]->item();
    }

    /**
     * @brief Forward pass for the multiplication operation.
     * 
     * Calculates the product of the two parent tensors and sets the result to the output tensor.
     * This method is used to compute the forward value of the multiplication operation in the computation graph.
     * 
     * The output tensor's value is set to the product of the values of the two parent tensors.
     * 
     * Example:
     *   If parent[0] = x and parent[1] = y, then output = x * y.
     */

    void forward() override {
        assert(this->output_ != nullptr);
        this->output_->set(this->parents[0]->item()*this->parents[1]->item());
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
    void backward() override {
        assert(this->output_ != nullptr);
        T tanh_x = this->output_->item();
        this->parents[0]->grad_ += this->output_->grad_ * (1 - tanh_x * tanh_x);
    }

    /**
     * @brief Forward pass of tanh operation
     * 
     * Calculates the forward operation of tanh currently for testing purposes
     */
    void forward() override{
        assert(this->output_ != nullptr);
        T data = this->parents[0]->item();
        T pos_exp = std::exp(data);
        T neg_exp = std::exp(-1*data);
        this->output_->set((pos_exp-neg_exp)/(pos_exp+neg_exp));
    }
};

}