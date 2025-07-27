#pragma once
#include <vector>
#include <cassert>
#include <numeric>
#include <random>
#include "tensor.hpp"

namespace backprop
{

/**
 * @brief types should only be float and double
 * multiple inputs one output
 */
template <typename T>
class Neuron{
    public:
    Neuron(int num_inputs){
        weights.resize(num_inputs);
    }

    /**
     * @brief randomly initializes the weights and bias of the neuron
     */
    void random_init(const T min = -1.0, const T max = 1.0){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);
        bias = dis(gen);
        for(Tensor<T>& weight : weights)
            weight = Tensor<T>(dis(gen));
    }

    Tensor<T> operator()(const std::vector<T>& x) const{
        assert(x.size() == weights.size() && "Number of inputs must match number of weights");
        return std::inner_product(x.begin(), x.end(), weights.begin(), bias);
    }

    void set_weights(const std::vector<T>& w){
        assert(w.size() == weights.size() && "Weight vector size must match number of weights");
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = Tensor<T>(w[i]);
        }
    }

    void set_bias(const T b){
        bias = b;
    }

    private:
    std::vector<Tensor<T>> weights;
    Tensor<T> bias;
};

/**
 * template should be float or double
 * multiple inputs multiple outputs
 */
template <typename T>
class Layer{
    public:
    Layer(int n_inputs, int n_outputs):num_inputs(n_inputs), num_outputs(n_outputs){
        neurons = std::vector<Neuron<T>>(n_outputs, Neuron<T>(n_inputs));
    }

    //TODO maybe make a setting values and biases function

    std::vector<Tensor<T>> operator()(const std::vector<T> x) const{
        assert(x.size() == num_inputs && "Number of inputs must match input dimension");
        std::vector<Tensor<T>> outputs;
        outputs.reserve(num_outputs);
        for(const Neuron<T>& neuron: neurons){
            outputs.push_back(neuron(x));
        }
        return outputs;
    }

    /**
     * @brief randomly initializes the neurons
     */
    void random_init(const T min = -1.0, const T max = 1.0){
        for(Neuron<T>& neuron: neurons){
            neuron.random_init();
        }
    }   

    private:
    const int num_inputs;
    const int num_outputs;
    std::vector<Neuron<T>> neurons;
};


}