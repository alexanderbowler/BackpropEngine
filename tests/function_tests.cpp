#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include "backprop/tensor.hpp"
#include "backprop/function.hpp"
#include "test_helpers.hpp"

TEST(FunctionTest, AddBasic){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::AddFunction<float> add_fn(t, t2);
    EXPECT_EQ(add_fn.parents[0], t.get_impl()) << "AddFunction should have t as parent";
    EXPECT_EQ(add_fn.parents[1], t2.get_impl()) << "AddFunction should have t2 as parent";
}

TEST(FunctionTest, AddBackward){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::AddFunction<float> add_fn(t, t2);

    // test backward
    backprop::Tensor<float> out(9.5);
    out.set_grad(1.0);
    add_fn.set_output_tensor(out);
    backward_function_test(add_fn);
}

TEST(FunctionTest, MultiplyBasic){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::MultiplyFunction<float> multiply_fn(t, t2);
    EXPECT_EQ(multiply_fn.parents[0], t.get_impl()) << "MultiplyFunction should have t as parent";
    EXPECT_EQ(multiply_fn.parents[1], t2.get_impl()) << "MultiplyFunction should have t2 as parent";
}

TEST(FunctionTest, MultiplyBackward){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::MultiplyFunction<float> multiply_fn(t, t2);
    // test backward
    backprop::Tensor<float> out(22.0);
    out.set_grad(1.0);
    multiply_fn.set_output_tensor(out);
    backward_function_test(multiply_fn);
}

TEST(FunctionTest, TanhBasic){
    backprop::Tensor<float> t(2.0);
    backprop::TanhFunction<float> tanh_fn(t);
    EXPECT_EQ(tanh_fn.parents[0], t.get_impl());
}

TEST(FunctionTest, TanhBackward){
    backprop::Tensor<float> t(2.0);
    backprop::TanhFunction<float> tanh_fn(t);

    //test backward
    backprop::Tensor<float> out(0.96402758);
    out.set_grad(1.0);
    // deriv of tanh(x) is 1-2.0^2 = -3.0, times outputis -6.0
    tanh_fn.set_output_tensor(out);
    backward_function_test(tanh_fn);
}

TEST(FunctionTest, ExpBasic){
    backprop::Tensor<float> t(2.0);
    backprop::ExpFunction<float> exp_fn(t);
    EXPECT_EQ(exp_fn.parents[0], t.get_impl());
    backprop::Tensor<float> out;
    exp_fn.set_output_tensor(out);
    exp_fn.forward();
    EXPECT_NEAR(out.item(), 7.389, 0.01);
}

TEST(FunctionTest, ExpBackward){
    backprop::Tensor<float> t(2.0);
    backprop::ExpFunction<float> exp_fn(t);
    backprop::Tensor<float> out;
    exp_fn.set_output_tensor(out);
    exp_fn.forward();
    out.set_grad(1.0);
    backward_function_test(exp_fn);
}

TEST(FunctionTest, PowFoward){
    backprop::Tensor<float> t(2.0);
    backprop::PowFunction<float, float> pow_fn(t, 3.0);
    EXPECT_EQ(pow_fn.parents[0], t.get_impl());
    backprop::Tensor<float> out;
    pow_fn.set_output_tensor(out);
    pow_fn.forward();
    EXPECT_FLOAT_EQ(out.item(), 8.0);
}

TEST(FunctionTest, PowBackward){
    backprop::Tensor<float> t(2.0);
    backprop::PowFunction<float, float> pow_fn(t, 3.0);
    backprop::Tensor<float> out;
    pow_fn.set_output_tensor(out);
    pow_fn.forward();
    out.set_grad(1.0);
    backward_function_test(pow_fn);
    EXPECT_FLOAT_EQ(t.grad(), 12.0);
}