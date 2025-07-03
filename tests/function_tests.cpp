#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include "backprop/tensor.hpp"
#include "backprop/function.hpp"
#include "test_helpers.hpp"

TEST(FunctionTest, AddFunctionTest){
    backprop::Tensor<float> t(4.0);
    backprop::Tensor<float> t2(5.5);
    backprop::AddFunction<float> add_fn(t, t2);
    EXPECT_EQ(add_fn.parents[0], t.get_impl()) << "AddFunction should have t as parent";
    EXPECT_EQ(add_fn.parents[1], t2.get_impl()) << "AddFunction should have t2 as parent";
}

// TEST(FunctionTest, AddFunctionBackward){
//     backprop::Tensor<float> t(4.0);
//     backprop::Tensor<float> t2(5.5);
//     backprop::AddFunction<float> add_fn(&t, &t2);

//     // test backward
//     backprop::Tensor<float> out(9.5);
//     out.grad_ = 1.0;
//     add_fn.set_output_tensor(&out);
//     backprop_function_test(add_fn);
//     // add_fn.backward();
//     // EXPECT_EQ(t.grad_, 1.5);
//     // EXPECT_EQ(t2.grad_, 1.5);
// }

// TEST(FunctionTest, MultiplyFunctionTest){
//     backprop::Tensor<float> t(4.0);
//     backprop::Tensor<float> t2(5.5);
//     backprop::MultiplyFunction<float> multiply_fn(&t, &t2);
//     EXPECT_EQ(multiply_fn.parents[0], &t) << "MultiplyFunction should have t as parent";
//     EXPECT_EQ(multiply_fn.parents[1], &t2) << "MultiplyFunction should have t2 as parent";

//     // test backward
//     backprop::Tensor<float> out(22.0);
//     out.grad_ = 1.0;
//     multiply_fn.set_output_tensor(&out);
//     backprop_function_test(multiply_fn);
//     // multiply_fn.backward();
//     // EXPECT_EQ(t.grad_, 11.0);
//     // EXPECT_EQ(t2.grad_, 8.0);
// }

// TEST(FunctionTest, TanhFunctionTest){
//     backprop::Tensor<float> t(2.0);
//     backprop::TanhFunction<float> tanh_fn(&t);
//     EXPECT_EQ(tanh_fn.parents[0], &t);

//     //test backward
//     // tanh(x) = 2.0
//     backprop::Tensor<float> out(0.96402758);
//     out.grad_ = 1.0;
//     // deriv of tanh(x) is 1-2.0^2 = -3.0, times outputis -6.0
//     tanh_fn.set_output_tensor(&out);
//     backprop_function_test(tanh_fn);
//     // tanh_fn.backward();
//     // EXPECT_EQ(t.grad_, result);
// }