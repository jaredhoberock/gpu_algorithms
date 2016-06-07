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
  T* ptr = nullptr;
  if(self.elect())
  {
    // allocate the storage
    std::size_t n = sizeof(T);
    ptr = reinterpret_cast<T*>(self.memory_resource().allocate<alignof(T)>(n));

    // construct the object
    ::new(ptr) T(std::forward<Args>(args)...);
  }

  using namespace agency::experimental;
  ptr = self.broadcast(ptr ? make_optional(ptr) : nullopt);

  return collective_ptr<T, concurrent_agent_deleter<ConcurrentAgent>>(ptr, concurrent_agent_deleter<ConcurrentAgent>(self));
}

