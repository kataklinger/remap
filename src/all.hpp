
// allocator library

#pragma once

#include <memory>

namespace all {

template<typename Alloc, typename Ty>
using rebind_alloc_t =
    typename std::allocator_traits<Alloc>::template rebind_alloc<Ty>;

class memory_pool {
private:
  static constexpr auto header_size{sizeof(char*)};

public:
  inline memory_pool() noexcept {
  }

  inline explicit memory_pool(std::size_t preallocated) {
    if (preallocated != 0) {
      extend(preallocated);
    }
  }

  inline memory_pool(memory_pool&& other) noexcept
      : total_allocated_{other.total_allocated_}
      , total_used_{other.total_used_}
      , current_size_{other.current_size_}
      , current_used_{other.current_used_}
      , current_{other.current_} {
    other.total_allocated_ = 0;
    other.total_used_ = 0;
    other.current_size_ = 0;
    other.current_used_ = 0;
    other.current_ = nullptr;
  }

  ~memory_pool() {
    for (auto current{current_}; current != nullptr;) {
      current -= header_size;

      auto next{*reinterpret_cast<char**>(current)};
      delete[] current;
      current = next;
    }
  }

  inline memory_pool& operator=(memory_pool&& rhs) noexcept {
    memory_pool tmp{std::move(rhs)};
    tmp.swap(*this);
    return *this;
  }

  memory_pool(memory_pool const&) = delete;
  memory_pool& operator=(memory_pool const&) = delete;

  [[nodiscard]] char* get(std::size_t size, std::size_t align) {
    if (size > current_size_ - current_used_) {
      extend(std::max(size + align, total_allocated_ >> 1));
    }

    current_used_ += reinterpret_cast<std::uintptr_t>(current_) % align;

    auto result{current_ + current_used_};
    current_used_ += size;
    total_used_ += size;

    return result;
  }

  [[nodiscard]] inline std::size_t total_used() const noexcept {
    return total_used_;
  }

  inline void swap(memory_pool& other) noexcept {
    std::swap(total_allocated_, other.total_allocated_);
    std::swap(total_used_, other.total_used_);
    std::swap(current_size_, other.current_size_);
    std::swap(current_used_, other.current_used_);
    std::swap(current_, other.current_);
  }

private:
  void extend(std::size_t size) {
    auto block{new char[size + header_size]};

    new (block) char*(current_);
    current_ = block + header_size;

    total_allocated_ += size;
    current_size_ = size;
    current_used_ = 0;
  }

private:
  std::size_t total_allocated_{0};
  std::size_t total_used_{0};

  std::size_t current_size_{0};
  std::size_t current_used_{0};

  char* current_{nullptr};
};

template<typename Ty>
class frame_allocator {
public:
  using value_type = Ty;

  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  inline explicit frame_allocator(memory_pool& pool) noexcept
      : pool_{&pool} {
  }

  template<typename Tx>
  inline frame_allocator(frame_allocator<Tx> const& other) noexcept
      : pool_{other.pool_} {
  }

  [[nodiscard]] inline value_type* allocate(std::size_t count) {
    return reinterpret_cast<value_type*>(
        pool_->get(count * sizeof(value_type), alignof(value_type)));
  }

  inline void deallocate(value_type* ptr, std::size_t count) const noexcept {
  }

  inline [[nodiscard]] bool
      operator==(frame_allocator const& rhs) const noexcept {
    return pool_ == rhs.pool_;
  }

  inline [[nodiscard]] bool
      operator!=(frame_allocator const& rhs) const noexcept {
    return !(*this == rhs);
  }

private:
  memory_pool* pool_;

  template<typename Tx>
  friend class frame_allocator;
};

template<typename Ty>
class memory_stack {
public:
  using allocator_type = frame_allocator<Ty>;

public:
  inline void prepare() {
    *current_ = {previous_->total_used() << 1};
  }

  inline void rotate() noexcept {
    std::swap(previous_, current_);
  }

  [[nodiscard]] inline allocator_type current() const noexcept {
    return allocator_type{*current_};
  }

  [[nodiscard]] inline allocator_type previous() const noexcept {
    return allocator_type{*previous_};
  }

private:
  memory_pool pools_[2];
  memory_pool* previous_{pools_};
  memory_pool* current_{pools_ + 1};
};

template<typename Ty>
class memory_swing {
public:
  using stack_type = memory_stack<Ty>;
  using allocator_type = typename stack_type::allocator_type;

public:
  inline explicit memory_swing(stack_type& stack) noexcept
      : stack_{&stack} {
    stack_->prepare();
  }

  inline ~memory_swing() {
    stack_->rotate();
  }

  [[nodiscard]] inline operator allocator_type() const noexcept {
    return get();
  }

  [[nodiscard]] inline allocator_type get() const noexcept {
    return stack_->current();
  }

private:
  stack_type* stack_;
};

} // namespace all
