#pragma once

#include <type_traits>
#include <agency/execution_agent.hpp>

template<class T, class CollectiveDeleter>
class collective_ptr 
{
  public:
    using element_type = T;
    using pointer      = element_type*;
    using deleter_type = CollectiveDeleter;

    __AGENCY_ANNOTATION
    collective_ptr(pointer ptr, const deleter_type& deleter)
      : ptr_(ptr),
        deleter_(deleter)
    {}

    __AGENCY_ANNOTATION
    ~collective_ptr()
    {
      deleter_(ptr_);
    }

    __AGENCY_ANNOTATION
    typename std::add_lvalue_reference<T>::type operator*() const
    {
      return *ptr_;
    }

    __AGENCY_ANNOTATION
    pointer operator->() const
    {
      return ptr_;
    }

  private:
    pointer ptr_;
    deleter_type deleter_;
};


template<class ConcurrentAgent>
class concurrent_agent_deleter
{
  public:
    using execution_agent_type = ConcurrentAgent;

    __AGENCY_ANNOTATION
    concurrent_agent_deleter(execution_agent_type& self)
      : self_(self)
    {}

    template<class T>
    __AGENCY_ANNOTATION
    void operator()(T* ptr)
    {
      if(self_.elect())
      {
        // destroy the object
        ptr->~T();

        // deallocate the storage
        self_.memory_resource().deallocate(ptr, sizeof(T));
      }

      self_.wait();
    }

  private:
    execution_agent_type& self_;
};


template<class T, class ConcurrentAgent, class... Args>
__AGENCY_ANNOTATION
collective_ptr<T, concurrent_agent_deleter<ConcurrentAgent>> make_collective(ConcurrentAgent& self, Args&&... args)
{
#ifdef __CUDA_ARCH__
  __shared__ T* shared_result;
  if(self.elect())
  {
    // allocate the storage
    std::size_t n = sizeof(T);
    T* ptr = reinterpret_cast<T*>(self.memory_resource().allocate<alignof(T)>(n));

    // construct the object
    ::new(ptr) T(std::forward<Args>(args)...);

    shared_result = ptr;
  }

  self.wait();
  return collective_ptr<T, concurrent_agent_deleter<ConcurrentAgent>>(shared_result, concurrent_agent_deleter<ConcurrentAgent>(self));
#else
  // XXX this seems kinda tough
  //     i think we need a static variable
  //     for this and we need to mediate access to it via a
  //     mutex
  // XXX it might be a better idea to build a small communication
  //     channel into concurrent_agent for these problems
  // XXX we could have a member function:
  //
  //     template<class T>
  //     T broadcast(const optional<T>& value)
  //
  //     Only one agent would present a non-empty value. The one non-empty
  //     value would be returned to the entire group.
  //
  //     There could be an internal buffer used for broadcasting, and its
  //     size could be implementation-defined.
  //
  //     Alternatively, the broadcasting buffer size could be configured
  //     by param_type.
  return nullptr;
#endif
}

