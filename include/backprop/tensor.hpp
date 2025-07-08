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
            m_pTensor = std::make_shared<TensorImpl<T>>(value);
            grad_fn->set_output_tensor(m_pTensor);
            m_pTensor->set_grad_fn(grad_fn);
        }

        /*
        @brief Constructor with grad_fn, for rvalues
        */
        // Tensor(T value, std::shared_ptr<Function<T>> grad_fn){
        // m_pTensor = std::make_shared<TensorImpl<T>>(value, grad_fn);
        // }

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

            /*
            @brief sets the gradient of the underlying tensor
            */
        void set_grad(T grad){
                m_pTensor->grad_ = grad;
        }

        /*
        @brief get gradient of undelrying tensor
        */
        const T grad() const {
            return m_pTensor->grad_;
        }

        /**
         * @brief computes the backward pass of the computational graph held by this tensor
         */
        void backward() const {
            m_pTensor->backward();
        }

        #ifdef UNIT_TEST
        /**
         * @brief test function to expose the grad_fn_ptr
         */
        const std::shared_ptr<Function<T>> get_func_ptr() const{
        return m_pTensor->grad_fn_ptr;
        }


        #endif
        
    protected:
        std::shared_ptr<TensorImpl<T>> m_pTensor;
};

template<typename T>
class TensorImpl: public std::enable_shared_from_this<TensorImpl<T>>{
    friend class Tensor<T>;
    friend class Function<T>;
    friend class AddFunction<T>;
    friend class MultiplyFunction<T>;
    friend class TanhFunction<T>;
    friend class ExpFunction<T>;
    friend class PowFunction<T, float>;
    friend class PowFunction<T, double>;
    friend class PowFunction<T, int>;
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
        @brief Default Constructor
        */
       TensorImpl() : data_(), shape_({}), grad_fn_ptr(nullptr), grad_(0.0){};

        /*
        @brief Sets the grad_function for the tensor implementation
        */
        void set_grad_fn(const std::shared_ptr<Function<T>>& grad_fn){
            grad_fn_ptr = grad_fn;
        }

        #ifdef UNIT_TEST
        template <typename U>
        friend void backward_function_test(backprop::Function<U>& fn);

        /**
         * @brief function for test which exposes data to public
         */
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
        void backward(){
            assert(grad_fn_ptr != nullptr);
            std::vector<TensorImpl<T>*> graph;
            build_topograph(graph, this);
            for(TensorImpl<T>* node: graph){
                node->grad_fn_ptr->backward();
            }
        }   

        

        std::shared_ptr<Function<T>> grad_fn_ptr;
        T grad_;
        T data_;
        std::vector<int> shape_;

        /**
         * @brief Builds a topological graph for backpropogation
         * @param graph graph to be built of TensorImpl pointers
         * @param t tensorImpl pointer of the beginning of the graph
         */
        void build_topograph(
            std::vector<TensorImpl<T>*>& graph,
            TensorImpl<T>* t
            ){
            std::unordered_set<TensorImpl<T>*> visited;
            build_topo_recursive(graph, t, visited);
            // This actually physically reverses the values in memory in future might just change 
            // the access order for more efficiency
            std::reverse(graph.begin(), graph.end());
            graph.shrink_to_fit();
        }

        /**
         * @brief recursive helper function to build the topological graph
         * only adds tensors to the graph which have a grad_fn_ptr, ie 
         * tensors thats have parents / a backwards function to call
         * @param graph graph which holds the tensorImpl pointers
         * @param t current tensor which we are looking at all parents from
         * @param visited set of TensorImpl pointers which have already been visited
         */
        void build_topo_recursive(
            std::vector<TensorImpl<T>*>& graph,
            TensorImpl<T>* t, 
            std::unordered_set<TensorImpl<T>*>& visited
        ){
            if(visited.count(t) || t->grad_fn_ptr == nullptr)
                    return;
            visited.insert(t);
            for(std::shared_ptr<TensorImpl<T>>& parent : t->grad_fn_ptr->parents){
                build_topo_recursive(graph, parent.get(), visited);
            }
            graph.push_back(t);
        }


};

#ifdef UNIT_TEST
/*
@brief function test helper
*/
template <typename T>
void backward_function_test(backprop::Function<T>& fn){
    T small_addition = 0.00001;
    T orig_output = fn.output_->item();
    // std::cout<<"Orig output: "<<orig_output<<"\n";

    fn.backward();
    for(std::shared_ptr<backprop::TensorImpl<T>> parent: fn.parents){
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
#endif

/**
 * @brief Add function for two tensors, uses AddFunction 
 */
template<typename T, typename U>
Tensor<T> operator+(const Tensor<T> lhs, const Tensor<U> rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot add tensors of two different data types");
    
    return Tensor<T>(lhs.item() + rhs.item(), std::make_shared<AddFunction<T>>(lhs, rhs));
}

/**
 * @brief adds a tensor to a constant
 */
template<typename T, typename U>
Tensor<T> operator+(Tensor<T>& lhs, U val){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot add tensors of two different data types");
    return lhs+Tensor<T>(val);
}

/**
 * @brief overloaded addition of constant with tensor
 */
template<typename T, typename U>
Tensor<T> operator+(U val, Tensor<T>& rhs){
    return rhs+val;
}

/**
 * @brief Multiply function for two tensors, creates and uses MultiplyFunction
 */
template<typename T, typename U>
Tensor<T> operator*(Tensor<T> lhs, Tensor<U> rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot multiply tensors of two different data types");
    
    return Tensor<T>(lhs.item() * rhs.item(), std::make_shared<MultiplyFunction<T>>(lhs, rhs));
}

/**
 * @brief Multiplication function for tensors with constants
 */
template<typename T, typename U>
Tensor<T> operator*(Tensor<T>& lhs, U val){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot multiply tensors of two different data types");
    // the temp tensor here creates a shared_ptr that is then used within the func later
    return lhs * Tensor<U>(val);
}

/**
 * @brief Multiplication function for tensors with constants
 */
template<typename T, typename U>
Tensor<T> operator*(U val, Tensor<T>& rhs){
    return rhs*val;
}

/**
 * @brief Tanh function for tensor
 */
template <typename T>
Tensor<T> tanh(Tensor<T>& t){
    T data = t.item();
    T pos_exp = std::exp(data);
    T neg_exp = std::exp(-1*data);
    return Tensor<T>((pos_exp-neg_exp)/(pos_exp+neg_exp), 
                    std::make_shared<TanhFunction<T>>(t));
}

/**
 * @brief Subtract function between two tensors
 */
template<typename T, typename U>
Tensor<T> operator-(Tensor<T> lhs, Tensor<U> rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot subtract tensors of two different data types");
    
    return lhs + (rhs * Tensor<T>(-1.0));
}

/**
 * @brief Subtract function tensor and constant
 */
template <typename T, typename U>
Tensor<T> operator-(Tensor<T> lhs, U val){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot subtract tensors of two different data types");
    return lhs + Tensor<T>(-1 * val);
}

/**
 * @brief Subtract function tensor and constant
 */
template <typename T, typename U>
Tensor<T> operator-(T val, Tensor<U> rhs){
    static_assert(std::is_same<T, U>::value, 
                    "Cannot subtract tensors of two different data types");
    return Tensor<T>(val) - rhs;
}

}