#include <gtest/gtest.h>
#include <iostream>
#include "backprop/tensor.hpp"
#include "backprop/function.hpp"
#include <cassert>
#include <typeinfo>


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

TEST(TEnsorTest, TanhForward){
    backprop::Tensor<float> t(1.0);
    backprop::Tensor<float> logits = tanh(t);
    float result = 0.76159;
    EXPECT_NEAR(logits.item(), result, 0.0001);
    EXPECT_NE(std::dynamic_pointer_cast<backprop::TanhFunction<float>>(logits.grad_fn_ptr),
    nullptr);
}

/*
Tests creating a chain of operations
In particular tests: ((4.0 * 5.5) + 2.0) * 3.0
Graph is 4.0  5.5
           \   /
             22.0  2.0
               \    /
                 24.0  3.0
                   \    /
                      72.0
*/
TEST(TensorTest, ChainOperations){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::Tensor<float> t3 = t*t2;
    backprop::Tensor<float> t4(2.0);
    backprop::Tensor<float> t5 = t3+t4;
    backprop::Tensor<float> t6(3.0);
    backprop::Tensor<float> t7 = t5*t6;
    EXPECT_EQ(t7.grad_fn_ptr->parents[0], &t5);
    EXPECT_EQ(t7.grad_fn_ptr->parents[1], &t6);
    EXPECT_EQ(t5.grad_fn_ptr->parents[0], &t3);
    EXPECT_EQ(t5.grad_fn_ptr->parents[1], &t4);
    EXPECT_EQ(t3.grad_fn_ptr->parents[0], &t);
    EXPECT_EQ(t3.grad_fn_ptr->parents[1], &t2);    
}

/*
Tests backprop on a chain of operations
In particular tests: ((4.0 * 5.5) + 2.0) * 3.0
Graph is 4.0  5.5
           \   /
             22.0  2.0
               \    /
                 24.0  3.0
                   \    /
                      72.0
*/
TEST(TensorTest, ChainBackpropogation){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::Tensor<float> t3 = t*t2;
    backprop::Tensor<float> t4(2.0);
    backprop::Tensor<float> t5 = t3+t4;
    backprop::Tensor<float> t6(3.0);
    backprop::Tensor<float> t7 = t5*t6;
    t7.grad_ = 1.0;
    t7.backward();
    EXPECT_EQ(t7.item(), 72.0);
    EXPECT_EQ(t6.grad_, 24.0);
    EXPECT_EQ(t5.grad_, 3.0);
    EXPECT_EQ(t4.grad_, 3.0);
    EXPECT_EQ(t3.grad_, 3.0);
    EXPECT_EQ(t2.grad_, 12.0);
    EXPECT_EQ(t.grad_, 16.5);
}

/*
Tests backprop on a chain of operations with multiple uses of same tensor
In particular tests: ((4.0 * 5.5) + (5.5 * -2.0)) * 3.0
Graph is 4.0  5.5  -2.0
           \   / \   /
             22.0  -11.0
               \    /
                 11.0  3.0
                   \    /
                      33.0
*/
TEST(TensorTest, DoubleUseBackpropogation){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::Tensor<float> t3 = t*t2;
    backprop::Tensor<float> t4(-2.0);
    backprop::Tensor<float> t5 = t2 * t4;
    backprop::Tensor<float> t6 = t3+t5;
    backprop::Tensor<float> t7(3.0);
    backprop::Tensor<float> t8 = t7*t6;
    t8.grad_ = 1.0;
    t8.backward();

    EXPECT_EQ(t3.item(), 22.0);
    EXPECT_EQ(t5.item(), -11.0);
    EXPECT_EQ(t6.item(), 11.0);
    EXPECT_EQ(t8.item(), 33.0);
    EXPECT_EQ(t7.grad_, 11.0);
    EXPECT_EQ(t6.grad_, 3.0);    
    EXPECT_EQ(t5.grad_, 3.0);    
    EXPECT_EQ(t4.grad_, 16.5);    
    EXPECT_EQ(t3.grad_, 3.0);    
    EXPECT_EQ(t2.grad_, 6.0);    
    EXPECT_EQ(t.grad_, 16.5);    
}



