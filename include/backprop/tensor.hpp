#pragma once
#include <vector>
#include <type_traits>
#include <typeinfo>
#include <algorithm>
#include <stack>
#include <unordered_set>
#include <iostream>
#include <cmath>

#include "function.hpp"
#include "constantRegistry.hpp"


namespace backprop{

template<typename T>
class Tensor{
    template <typename> friend class TensorTest;
    public:
        Tensor(T value): data_(value), shape_({}), grad_(0.0) {
            grad_fn_ptr = nullptr;
        }
        Tensor(T value, std::shared_ptr<Function<T>> grad_fn): 
            data_(value), shape_({}), grad_fn_ptr(grad_fn), grad_(0.0) {
                grad_fn->set_output_tensor(this);
            }

        const T item() const{
            return data_;
        }

        void set(T new_data){
            data_ = new_data;
        }

        const std::vector<int>& shape() const{
            return shape_;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){

            std::string output = "Tensor<" + std::string(typeid(T).name()) + ">(";
            os<<output;
            std::string shape = "";
            for(int dimension: tensor.shape()){
                shape += std::to_string(dimension) + ", ";
            }
            shape[shape.length()-2] = ')';
            os<<shape;
            std::string val = "{" + std::to_string(tensor.item()) + "}\n";
            os<<val;
            return os;
        }

        
        #ifdef UNIT_TEST
        const T get_data() const{
            return this->data_;
        }
        #endif

        // Calls the corresponding backward function
        // REQUIRES: The gradient for this tensor is set
        void backward(){
            assert(grad_fn_ptr != nullptr);
            std::vector<Tensor<T>*> graph;
            build_topograph(graph, this);
            for(Tensor<T>* node: graph){
                node->grad_fn_ptr->backward();
            }
        }   

        

        std::shared_ptr<Function<T>> grad_fn_ptr;
        T grad_;
        friend class TensorTestAccess;  // Add this line
    protected:
        T data_;
        std::vector<int> shape_;

        // Builds a topological graph for backpropogation
        void build_topograph(
            std::vector<Tensor<T>*>& graph,
            Tensor<T>* t
            ){
            std::unordered_set<Tensor<T>*> visited;
            build_topo_recursive(graph, t, visited);
            // This actually physically reverses the values in memory in future might just change 
            // the access order for more efficiency
            std::reverse(graph.begin(), graph.end());
            graph.shrink_to_fit();
        }

        // recursive helper function to build the topological graph
        // only adds tensors to the graph which have a grad_fn_ptr, ie 
        // tensors thats have parents / a backwards function to call
        void build_topo_recursive(
            std::vector<Tensor<T>*>& graph,
            Tensor<T>* t, 
            std::unordered_set<Tensor<T>*>& visited
        ){
            if(visited.count(t) || t->grad_fn_ptr == nullptr)
                    return;
            visited.insert(t);
            for(Tensor<T>* parent : t->grad_fn_ptr->parents){
                build_topo_recursive(graph, parent, visited);
            }
            graph.push_back(t);
        }


};

template<typename T, typename U>
Tensor<T> operator+(Tensor<T>& lfs, Tensor<U>& rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot add tensors of two different data types");
    
    return Tensor<T>(lfs.item() + rhs.item(), std::make_shared<AddFunction<T>>(&lfs, &rhs));
}

template<typename T, typename U>
Tensor<T> operator+(Tensor<T>& lfs, U val){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot add tensors of two different data types");
    Tensor<T>* p_rhs = ConstantRegistry<T>::get_constant(val);
    return lfs+(*p_rhs);
}

template<typename T, typename U>
Tensor<T> operator+(U val, Tensor<T>& rhs){
    return rhs+val;
}

template<typename T, typename U>
Tensor<T> operator*(Tensor<T>& lfs, Tensor<U>& rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot multiply tensors of two different data types");
    
    return Tensor<T>(lfs.item() * rhs.item(), std::make_shared<MultiplyFunction<T>>(&lfs, &rhs));
}

template<typename T, typename U>
Tensor<T> operator*(Tensor<T>& lfs, U val){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot multiply tensors of two different data types");
    Tensor<T>* p_rhs = ConstantRegistry<T>::get_constant(val);
    return lfs* (*p_rhs);
}

template<typename T, typename U>
Tensor<T> operator*(U val, Tensor<T>& rhs){
    return rhs*val;
}

template <typename T>
Tensor<T> tanh(Tensor<T>& t){
    T data = t.item();
    T pos_exp = std::exp(data);
    T neg_exp = std::exp(-1*data);
    return Tensor<T>((pos_exp-neg_exp)/(pos_exp+neg_exp), 
                    std::make_shared<TanhFunction<T>>(&t));
}

template<typename T, typename U>
Tensor<T> operator-(Tensor<T>& lfs, Tensor<U>& rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot subtract tensors of two different data types");
    
    return lfs + (rhs * static_cast<T>(-1.0));
}

}