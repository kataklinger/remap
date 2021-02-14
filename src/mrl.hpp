
#pragma once

#include <concepts>
#include <vector>

namespace mrl {
using size_type = std::size_t;

template<typename Ty>
class matrix {
public:
  using value_type = Ty;

public:
  inline matrix(size_type width, size_type height)
      : width_{width}
      , height_{height} {
    data_.resize(width * height);
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
    matrix<target_type> output{width_, height_};

    auto out{output.data()};
    for (auto cur{data_.data()}, last{data_.data() + data_.size()}; cur < last;
         ++cur) {
      *(out++) = std::invoke(convert, *cur);
    }

    return output;
  }

  matrix
      crop(size_type left, size_type right, size_type top, size_type bottom) {
    auto stride{left + right};
    matrix output{width_ - stride, height_ - top - bottom};

    for (auto src{data_.data() + top * width_ + left},
         dst{output.data()},
         last{output.end()};
         dst < last;
         src += stride) {

      for (auto end{src + output.width()}; src < end; ++src, ++dst) {
        *dst = *src;
      }
    }

    return output;
  }

private:
  size_type width_;
  size_type height_;

  std::vector<value_type> data_;
};

} // namespace mrl
