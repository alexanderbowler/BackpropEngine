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

template <typename T>
class TensorImpl;

template <typename T>
class Tensor{
    public:
        /* 
        @brief Tensor Constructor by value, makes the shared ptr
        */
        Tensor(T value){
            m_pTensor = std::make_shared<TensorImpl<T>>(value);
        }
        /* 
        @brief Tensor Copy Constructor
        */
        Tensor(const Tensor<T>& t) : m_pTensor(t.m_pTensor){};

        /*
        @brief default constructor
        */
       Tensor(){
        m_pTensor = std::make_shared<TensorImpl<T>>();
       }

       // TODO: Create assignment operators and overload with assignment of just values as well

        /*
        @brief Constructor of a tensor with the corresponding function that created it
        version for lvalues
        */
        Tensor(T value, const std::shared_ptr<Function<T>>& grad_fn){
            m_pTensor = std::make_shared<TensorImpl<T>>(value, grad_fn);
        }

        /*
        @brief Constructor with grad_fn, for rvalues
        */
        Tensor(T value, std::shared_ptr<Function<T>> grad_fn){
        m_pTensor = std::make_shared<TensorImpl<T>>(value, grad_fn);
        }

        /*
        @brief Getter for the TensorImpl
        */
        std::shared_ptr<TensorImpl<T>> get_impl() const{
        return m_pTensor;
        }

        /*
        @brief gets the value the tensor holds
        */
        const T item() const{
        return m_pTensor->item();
        }

        /*
        @brief sets the value within the tensor
        */
        void set(T new_data){
        m_pTensor->set(new_data);
        }

        /*
        @brief gets the shape of the tensor
        */
        const std::vector<int>& shape() const {
            return m_pTensor->shape(); 
        }

        /*
        @brief cout operator overload for printing
        */
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){

                os << tensor.m_pTensor->to_string();
                return os;
        }
        
    protected:
        std::shared_ptr<TensorImpl<T>> m_pTensor;
};

template<typename T>
class TensorImpl{
    friend class Tensor<T>;
    template <typename> friend class TensorTest;
    public:
        /*
        @brief stringifies the tensor for printing
        */
        const std::string to_string() const {
            std::string output = "Tensor<" + std::string(typeid(T).name()) + ">(";
            for(int dimension: shape_){
                output += std::to_string(dimension) + ", ";
            }
            if(shape_.size() == 0)
                output += ')';
            else
                output[output.length()-2] = ')';
            output += " {" + std::to_string(data_) + "}\n";
            return output;
        }
        /*
        @brief Basic Constructor with value
        */
        TensorImpl(T value): data_(value), shape_({}), grad_(0.0) {
            grad_fn_ptr = nullptr;
        }

        /*
        @brief Constructor with value and grad_fn
        */
        TensorImpl(T value, const std::shared_ptr<Function<T>>& grad_fn): 
            data_(value), shape_({}), grad_fn_ptr(grad_fn), grad_(0.0) {
                grad_fn->set_output_tensor(this);
        }

        /*
        @brief Default Constructor
        */
       TensorImpl() : data_(), shape_({}), grad_fn_ptr(nullptr), grad_(0.0){};

        #ifdef UNIT_TEST
        const T get_data() const{
            return this->data_;
        }
        #endif

    protected:

        const T item() const{
            return data_;
        }

        void set(T new_data){
            data_ = new_data;
        }

        const std::vector<int>& shape() const{
            return shape_;
        }

        // Calls the corresponding backward function
        // REQUIRES: The gradient for this tensor is set
        // void backward(){
        //     assert(grad_fn_ptr != nullptr);
        //     std::vector<TensorImpl<T>*> graph;
        //     build_topograph(graph, this);
        //     for(TensorImpl<T>* node: graph){
        //         node->grad_fn_ptr->backward();
        //     }
        // }   

        

        std::shared_ptr<Function<T>> grad_fn_ptr;
        T grad_;
    protected:
        T data_;
        std::vector<int> shape_;

        // Builds a topological graph for backpropogation
        // void build_topograph(
        //     std::vector<Tensor<T>*>& graph,
        //     Tensor<T>* t
        //     ){
        //     std::unordered_set<Tensor<T>*> visited;
        //     build_topo_recursive(graph, t, visited);
        //     // This actually physically reverses the values in memory in future might just change 
        //     // the access order for more efficiency
        //     std::reverse(graph.begin(), graph.end());
        //     graph.shrink_to_fit();
        // }

        // // recursive helper function to build the topological graph
        // // only adds tensors to the graph which have a grad_fn_ptr, ie 
        // // tensors thats have parents / a backwards function to call
        // void build_topo_recursive(
        //     std::vector<Tensor<T>*>& graph,
        //     Tensor<T>* t, 
        //     std::unordered_set<Tensor<T>*>& visited
        // ){
        //     if(visited.count(t) || t->grad_fn_ptr == nullptr)
        //             return;
        //     visited.insert(t);
        //     for(Tensor<T>* parent : t->grad_fn_ptr->parents){
        //         build_topo_recursive(graph, parent, visited);
        //     }
        //     graph.push_back(t);
        // }


};

template<typename T, typename U>
Tensor<T> operator+(Tensor<T>& lfs, Tensor<U>& rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot add tensors of two different data types");
    
    return Tensor<T>(lfs.item() + rhs.item(), std::make_shared<AddFunction<T>>(&lfs, &rhs));
}

// template<typename T, typename U>
// Tensor<T> operator+(Tensor<T>& lfs, U val){
//     static_assert(std::is_same<T, U>::value, 
//                     "Cannot add tensors of two different data types");
//     Tensor<T>* p_rhs = ConstantRegistry<T>::get_constant(val);
//     return lfs+(*p_rhs);
// }

// template<typename T, typename U>
// Tensor<T> operator+(U val, Tensor<T>& rhs){
//     return rhs+val;
// }

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