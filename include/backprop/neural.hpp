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


}