
// matrix library

#pragma once

#include "cdt.hpp"

#include <concepts>
#include <tuple>
#include <vector>

namespace mrl {
using size_type = std::size_t;
using point_t = cdt::point<size_type>;
using dimensions_t = cdt::dimensions<size_type>;
using limits_t = cdt::limits<size_type>;
using region_t = cdt::region<size_type>;

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
      : data_{alloc} {
  }

  inline matrix(dimensions_t dimensions, allocator_type const& alloc)
      : dimensions_{dimensions}
      , data_{alloc} {
    data_.resize(dimensions.area());
  }

  inline matrix(dimensions_t dimensions,
                const value_type& value,
                allocator_type const& alloc)
      : dimensions_{dimensions}
      , data_{alloc} {
    data_.resize(dimensions.area(), value);
  }

  inline matrix(dimensions_t dimensions)
      : matrix(dimensions, allocator_type{}) {
  }

  inline matrix(dimensions_t dimensions, const value_type& value)
      : matrix(dimensions, value, allocator_type{}) {
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
    return dimensions_.width_;
  }

  [[nodiscard]] inline size_type height() const noexcept {
    return dimensions_.height_;
  }

  [[nodiscard]] inline size_type size() const noexcept {
    return data_.size();
  }

  [[nodiscard]] inline dimensions_t const& dimensions() const noexcept {
    return dimensions_;
  }

  template<typename Fn>
  [[nodiscard]] auto map(Fn convert) const
      requires std::invocable<Fn, value_type const&> {
    using target_type = std::invoke_result_t<Fn, const value_type&>;
    matrix<target_type> output{dimensions_, data_.get_allocator()};

    auto out{output.data()};
    for (auto cur{data_.data()}, last{data_.data() + data_.size()}; cur < last;
         ++cur) {
      *(out++) = std::invoke(convert, *cur);
    }

    return output;
  }

  [[nodiscard]] matrix crop(region_t region) const {
    auto margins{region.margins()};

    matrix output{
        {dimensions_.width_ - margins.x_, dimensions_.height_ - margins.y_},
        data_.get_allocator()};

    auto nwidth{output.width()};

    auto src{data() + region.top_ * dimensions_.width_ + region.left_};
    for (auto dst{output.data()}, last{output.end()}; dst < last;
         src += dimensions_.width_, dst += nwidth) {
      std::copy(src, src + nwidth, dst);
    }

    return output;
  }

  [[nodiscard]] matrix extend(region_t region) const {
    auto margins{region.margins()};

    matrix output{
        {dimensions_.width_ + margins.x_, dimensions_.height_ + margins.y_},
        data_.get_allocator()};

    auto nwidth{output.width()};

    auto dst{output.data() + region.top_ * nwidth + region.left_};
    for (auto src{data()}, last{end()}; src < last;
         src += dimensions_.width_, dst += nwidth) {
      std::copy(src, src + dimensions_.width_, dst);
    }

    return output;
  }

  inline allocator_type get_allocator() const noexcept {
    return data_.get_allocator();
  }

private:
  dimensions_t dimensions_{};

  std::vector<value_type, allocator_type> data_;
};

} // namespace mrl
