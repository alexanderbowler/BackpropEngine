#include <gtest/gtest.h>
#include <iostream>
#include "backprop/tensor.hpp"
#include "backprop/function.hpp"
#include "backprop/constantRegistry.hpp"
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
  EXPECT_EQ(t.get_impl()->get_data(), 5.5);
}

/*
@brief tests copy constructor and that copies of Tensor wrapper keep the same underlying object
*/
TEST(TensorTest, CopyTest){
  backprop::Tensor<float> t(1.5);
  backprop::Tensor<float> copy = t; //calls copy constructor
  EXPECT_EQ(copy.item(), 1.5);
  EXPECT_EQ(t.get_impl(), copy.get_impl());
}

/*
@brief tests the default constructor
*/
TEST(TensorTest, DefaultConstructor){
  backprop::Tensor<float> t;
  EXPECT_EQ(t.item(), 0.);
}

/*
@brief tests adding two tensors
*/
TEST(TensorTest, Add){
  backprop::Tensor<float> t(4.0);
  backprop::Tensor<float> t2(5.5);
  backprop::Tensor<float> sum = t+t2;
  EXPECT_EQ(sum.item(), 9.5);
  std::shared_ptr<backprop::Function<float>> grad_fn_ptr = sum.get_func_ptr();
  EXPECT_NE(grad_fn_ptr, nullptr);
  EXPECT_EQ(grad_fn_ptr->output_, sum.get_impl().get());
  EXPECT_EQ(grad_fn_ptr->parents[0], t.get_impl());
  EXPECT_EQ(grad_fn_ptr->parents[1], t2.get_impl());
}

TEST(TensorTest, AddBackward){
  backprop::Tensor<float> t(4.0);
  backprop::Tensor<float> t2(5.5);
  backprop::Tensor<float> sum = t+t2;
  sum.set_grad(1.0);
  sum.backward();
  EXPECT_EQ(t.grad(), 1.0);
  EXPECT_EQ(t2.grad(), 1.0);
}

TEST(TensorTest, Multiply){
  backprop::Tensor<float> t(4.0);
  backprop::Tensor<float> t2(5.5);
  backprop::Tensor<float> product = t*t2;
  EXPECT_EQ(product.item(), 22.0);
  std::shared_ptr<backprop::Function<float>> grad_fn_ptr = product.get_func_ptr();
  EXPECT_NE(grad_fn_ptr, nullptr);
  EXPECT_EQ(grad_fn_ptr->output_, product.get_impl().get());
  EXPECT_EQ(grad_fn_ptr->parents[0], t.get_impl());
  EXPECT_EQ(grad_fn_ptr->parents[1], t2.get_impl());
}

TEST(TensorTest, MultiplyBackward){
  backprop::Tensor<float> t(4.0);
  backprop::Tensor<float> t2(5.5);
  backprop::Tensor<float> product = t*t2;
  product.set_grad(1.0);
  product.backward();
  EXPECT_EQ(t.grad(), 5.5);
  EXPECT_EQ(t2.grad(), 4.0);
}

TEST(TensorTest, TanhForward){
  backprop::Tensor<float> t(1.0);
  backprop::Tensor<float> logits = tanh(t);
  float result = 0.76159;
  EXPECT_NEAR(logits.item(), result, 0.0001);
  std::shared_ptr<backprop::Function<float>> grad_fn_ptr = logits.get_func_ptr();
  EXPECT_NE(grad_fn_ptr, nullptr);
  EXPECT_EQ(grad_fn_ptr->output_, logits.get_impl().get());
}

TEST(TensorTest, TanhBackward){
  backprop::Tensor<float> t(1.0);
  backprop::Tensor<float> logits = tanh(t);
  float result = 0.76159;
  logits.set_grad(1.0);
  logits.backward();
  EXPECT_NEAR(t.grad(), 0.419974, 0.001);
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
  EXPECT_EQ(t7.get_func_ptr()->parents[0], t5.get_impl());
  EXPECT_EQ(t7.get_func_ptr()->parents[1], t6.get_impl());
  EXPECT_EQ(t5.get_func_ptr()->parents[0], t3.get_impl());
  EXPECT_EQ(t5.get_func_ptr()->parents[1], t4.get_impl());
  EXPECT_EQ(t3.get_func_ptr()->parents[0], t.get_impl());
  EXPECT_EQ(t3.get_func_ptr()->parents[1], t2.get_impl());    
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
  t7.set_grad(1.0);
  t7.backward();
  EXPECT_EQ(t7.item(), 72.0);
  EXPECT_EQ(t6.grad(), 24.0);
  EXPECT_EQ(t5.grad(), 3.0);
  EXPECT_EQ(t4.grad(), 3.0);
  EXPECT_EQ(t3.grad(), 3.0);
  EXPECT_EQ(t2.grad(), 12.0);
  EXPECT_EQ(t.grad(), 16.5);
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
  t8.set_grad(1.0);
  t8.backward();

  EXPECT_EQ(t3.item(), 22.0);
  EXPECT_EQ(t5.item(), -11.0);
  EXPECT_EQ(t6.item(), 11.0);
  EXPECT_EQ(t8.item(), 33.0);
  EXPECT_EQ(t7.grad(), 11.0);
  EXPECT_EQ(t6.grad(), 3.0);    
  EXPECT_EQ(t5.grad(), 3.0);    
  EXPECT_EQ(t4.grad(), 16.5);    
  EXPECT_EQ(t3.grad(), 3.0);    
  EXPECT_EQ(t2.grad(), 6.0);    
  EXPECT_EQ(t.grad(), 16.5);    
}

TEST(TensorTest, MultiplyWithConstants){
  backprop::Tensor<float> t(1.5);
  backprop::Tensor<float> res = t*2.0f;
  EXPECT_EQ(res.item(), 3.0);
  EXPECT_EQ(res.get_func_ptr()->parents[1]->get_data(), 2.0f);
  res.set_grad(1.0);
  res.backward();
  EXPECT_EQ(t.grad(), 2.0);
  backprop::Tensor<float> res2 = 3.0f*t;
  EXPECT_EQ(res2.item(), 4.5);
}

TEST(TensorTest, AddWithConstants){
  backprop::Tensor<float> t(1.5);
  backprop::Tensor<float> res = t+2.0f;
  EXPECT_EQ(res.item(), 3.5);
  res.set_grad(1.0);
  res.backward();
  EXPECT_EQ(t.grad(), 1.0);
  backprop::Tensor<float> res2 = 3.0f+t;
  EXPECT_EQ(res2.item(), 4.5);
  EXPECT_TRUE(res2.get_func_ptr()->parents[1]->get_data() == 3.0 || 
  res2.get_func_ptr()->parents[0]->get_data() == 3.0);
}

TEST(TensorTest, Subtract){
  backprop::Tensor<float> t(2.5);
  backprop::Tensor<float> t2(1.5);
  backprop::Tensor<float> res = t-t2;
  EXPECT_EQ(res.item(), 1.0f);
  res.set_grad(1.5f);
  res.backward();
  EXPECT_EQ(t.grad(), 1.5f);
  EXPECT_EQ(t2.grad(), -1.5f);
}

TEST(TensorTest, SubtractWithConstants){
  backprop::Tensor<float> t(1.5); 
  backprop::Tensor<float> res = t-2.0f;
  EXPECT_EQ(res.item(), -0.5);
  res.set_grad(1.5);
  res.backward();
  EXPECT_EQ(t.grad(), 1.5);
  backprop::Tensor<float> res2 = 3.0f-t;
  EXPECT_EQ(res2.item(), 1.5);
  t.set_grad(0.0);
  res2.set_grad(1.5);
  res2.backward();
  EXPECT_EQ(t.grad(), -1.5);
  EXPECT_TRUE(res2.get_func_ptr()->parents[0]->get_data() == 3.0 || 
  res2.get_func_ptr()->parents[1]->get_data() == 3.0);
}