#include <gtest/gtest.h>
#include <iostream>
#include "backprop/tensor.hpp"
#include "backprop/function.hpp"

TEST(FunctionTest, AddFunctionTest){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::AddFunction<float> add_fn(&t, &t2);
    EXPECT_EQ(add_fn.parents[0], &t) << "AddFunction should have t as parent";
    EXPECT_EQ(add_fn.parents[1], &t2) << "AddFunction should have t2 as parent";
}

TEST(FunctionTest, MultiplyFunctionTest){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::MultiplyFunction<float> multiply_fn(&t, &t2);
    EXPECT_EQ(multiply_fn.parents[0], &t) << "MultiplyFunction should have t as parent";
    EXPECT_EQ(multiply_fn.parents[1], &t2) << "MultiplyFunction should have t2 as parent";
}