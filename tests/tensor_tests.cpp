#include <gtest/gtest.h>
#include <iostream>
#include "backprop/tensor.hpp"


TEST(TensorTest, ShapeIsCorrect){
    backprop::Tensor<float> t(4.0);
    std::vector<int> expected = {};
    EXPECT_EQ(t.shape(), expected);
}

TEST(TensorTest, BasicValueTest){
    backprop::Tensor<float> t(5.5);
    EXPECT_EQ(t.item(), 5.5);
    EXPECT_EQ(t.get_data(), 5.5);
}

TEST(TensorTest, AddTensorTest){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::Tensor<float> sum = t+t2;
    EXPECT_EQ(sum.item(), 9.5);
    EXPECT_EQ(sum.grad_fn_ptr->parents[0], &t);
    EXPECT_EQ(sum.grad_fn_ptr->parents[1], &t2);
}

TEST(TensorTest, AddBackwardTest){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::Tensor<float> sum = t+t2;
    sum.grad_ = 1.0;
    sum.backward();
    EXPECT_EQ(t.grad_, 1.0);
    EXPECT_EQ(t2.grad_, 1.0);
}

TEST(TensorTest, MultiplyTest){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::Tensor<float> product = t*t2;
    EXPECT_EQ(product.item(), 22.0);
    EXPECT_EQ(product.grad_fn_ptr->parents[0], &t);
    EXPECT_EQ(product.grad_fn_ptr->parents[1], &t2);
}

TEST(TensorTest, MultiplyBackwardTest){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::Tensor<float> product = t*t2;
    product.grad_ = 1.0;
    product.backward();
    EXPECT_EQ(t.grad_, 5.5);
    EXPECT_EQ(t2.grad_, 4.0);
}


