
#pragma once

#include <concepts>
#include <vector>

namespace mrl {
using size_type = std::size_t;

template<typename Ty, typename Alloc = std::allocator<Ty>>
class matrix {
public:
  using value_type = Ty;
  using allocator_type = Alloc;

public:
  inline matrix() noexcept
      : matrix(allocator_type{}) {
  }

  inline explicit matrix(allocator_type const& alloc) noexcept
      : width_{0}
      , height_{0}
      , data_{alloc} {
  }

  inline matrix(size_type width, size_type height, allocator_type const& alloc)
      : width_{width}
      , height_{height}
      , data_{alloc} {
    data_.resize(width * height);
  }

  inline matrix(size_type width, size_type height)
      : matrix(width, height, allocator_type{}) {
  }

  [[nodiscard]] inline value_type& operator[](size_type index) noexcept {
    return data_[index];
  }

  [[nodiscard]] inline value_type const&
      operator[](size_type index) const noexcept {
    return data_[index];
  }

  [[nodiscard]] inline value_type* data() noexcept {
    return data_.data();
  }

  [[nodiscard]] inline const value_type* data() const noexcept {
    return data_.data();
  }

  [[nodiscard]] inline value_type* end() noexcept {
    return data_.data() + data_.size();
  }

  [[nodiscard]] inline const value_type* end() const noexcept {
    return data_.data() + data_.size();
  }

  [[nodiscard]] inline size_type width() const noexcept {
    return width_;
  }

  [[nodiscard]] inline size_type height() const noexcept {
    return height_;
  }

  [[nodiscard]] inline size_type size() const noexcept {
    return data_.size();
  }

  template<typename Fn>
  [[nodiscard]] auto map(Fn convert) const
      requires std::invocable<Fn, value_type const&> {
    using target_type = std::invoke_result_t<Fn, const value_type&>;
    matrix<target_type> output{width_, height_, data_.get_allocator()};

    auto out{output.data()};
    for (auto cur{data_.data()}, last{data_.data() + data_.size()}; cur < last;
         ++cur) {
      *(out++) = std::invoke(convert, *cur);
    }

    return output;
  }

  matrix
      crop(size_type left, size_type right, size_type top, size_type bottom) {
    matrix output{
        width_ - left - right, height_ - top - bottom, data_.get_allocator()};

    auto nwidth{output.width()};

    for (auto src{data() + top * width_ + left},
         dst{output.data()},
         last{output.end()};
         dst < last;
         src += width_, dst += nwidth) {

      std::copy(src, src + nwidth, dst);
    }

    return output;
  }

  matrix
      extend(size_type left, size_type right, size_type top, size_type bottom) {
    matrix output{
        width_ + left + right, height_ + top + bottom, data_.get_allocator()};

    auto nwidth{output.width()};

    for (auto src{data()},
         dst{output.data() + top * nwidth + left},
         last{end()};
         src < last;
         src += width_, dst += nwidth) {

      std::copy(src, src + width_, dst);
    }

    return output;
  }

  inline allocator_type get_allocator() const noexcept {
    return data_.get_allocator();
  }

private:
  size_type width_;
  size_type height_;

  std::vector<value_type, allocator_type> data_;
};

} // namespace mrl
