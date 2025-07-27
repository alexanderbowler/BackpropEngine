#include <gtest/gtest.h>
#include "backprop/neural.hpp"
#include "backprop/tensor.hpp"

TEST(NeuronTest, NeuronForward){
    backprop::Neuron<float> neuron(2);
    neuron.set_weights({0.5, 2.0});
    neuron.set_bias(-1.0);
    std::vector<float> x = {1.0, 2.0};
    backprop::Tensor<float> result = neuron(x);
    EXPECT_EQ(result.item(), 3.5);
}

TEST(NeuronTest, RandomInit){
    backprop::Neuron<float> neuron(2);
    neuron.random_init();
    std::vector<float> x = {1.0, 2.0};
    backprop::Tensor<float> result = neuron(x);
    EXPECT_NE(result.item(), 0.0);
}

TEST(LayerTest, LayerForward){
    backprop::Layer<float> layer(3,4);    
    std::vector<float> x = {1.0, 2.0, 3.0};
    std::vector<backprop::Tensor<float>> result = layer(x);
    EXPECT_EQ(result.size(), 4);
}

TEST(LayerTest, RandomInit){
    backprop::Layer<float> layer(3,1);
    layer.random_init();
    std::vector<float> x = {1.0, 2.0};
    std::vector<backprop::Tensor<float>> result = layer(x);
    EXPECT_NE(result[0].item(), 0.0);
}