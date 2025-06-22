#pragma once
#include "backprop/tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>


template <typename T>
void backprop_function_test(backprop::Function<T>& fn){
    T small_addition = 0.00001;
    T orig_output = fn.output_->item();
    // std::cout<<"Orig output: "<<orig_output<<"\n";

    fn.backward();
    for(backprop::Tensor<T>* parent: fn.parents){
        T orig_parent_val = parent->item();
        parent->set(orig_parent_val + small_addition);
        fn.forward();
        // std::cout<<"parent "<<parent->item()<<"\n";
        // std::cout<<"Modded output: "<<fn.output_->item()<<"\n";
        T gradient = (fn.output_->item() - orig_output) / small_addition;
        EXPECT_NEAR(parent->grad_, gradient, 0.05);
        parent->set(orig_parent_val);
    }
}