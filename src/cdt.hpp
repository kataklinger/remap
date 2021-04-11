
#pragma once

#include <concepts>
#include <cstdint>
#include <tuple>

namespace cdt {
template<std::integral Ty>
struct point {
  using value_type = Ty;

  value_type x_{};
  value_type y_{};

  inline point& operator+=(point const& rhs) noexcept {
    x_ += rhs.x_;
    y_ += rhs.y_;

    return *this;
  }

  inline point& operator-=(point const& rhs) noexcept {
    x_ -= rhs.x_;
    y_ -= rhs.y_;

    return *this;
  }

  template<std::integral Tx>
  [[nodiscard]] inline explicit operator point<Tx>() const noexcept {
    return {static_cast<Tx>(x_), static_cast<Tx>(y_)};
  }
};

template<std::size_t Idx, std::integral Ty>
Ty get(point<Ty> const& pt) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return pt.x_;
  }
  else {
    return pt.y_;
  }
}

template<std::size_t Idx, std::integral Ty>
Ty& get(point<Ty>& pt) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return pt.x_;
  }
  else {
    return pt.y_;
  }
}

template<std::size_t Idx, std::integral Ty>
Ty&& get(point<Ty>&& pt) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return std::move(pt.x_);
  }
  else {
    return std::move(pt.y_);
  }
}

template<std::integral Ty>
[[nodiscard]] inline point<Ty> operator+(point<Ty> const& lhs,
                                         point<Ty> const& rhs) noexcept {
  auto tmp{lhs};
  tmp += rhs;
  return tmp;
}

template<std::integral Ty>
[[nodiscard]] inline point<Ty> operator-(point<Ty> const& lhs,
                                         point<Ty> const& rhs) noexcept {
  auto tmp{lhs};
  tmp -= rhs;
  return tmp;
}

template<std::integral Ty>
[[nodiscard]] inline point<Ty> operator-(point<Ty> const& rhs) noexcept {
  return {-rhs.x_, -rhs.y_};
}

template<std::integral Ty>
[[nodiscard]] inline bool operator==(point<Ty> const& lhs,
                                     point<Ty> const& rhs) noexcept {
  return lhs.x_ == rhs.x_ && lhs.y_ == rhs.y_;
}

template<std::integral Ty>
[[nodiscard]] inline bool operator!=(point<Ty> const& lhs,
                                     point<Ty> const& rhs) noexcept {
  return !(lhs == rhs);
}

template<std::unsigned_integral Ty>
struct dimensions {
  using value_type = Ty;

  [[nodicard]] inline value_type area() const noexcept {
    return width_ * height_;
  }

  value_type width_{};
  value_type height_{};
};

template<std::size_t Idx, std::unsigned_integral Ty>
Ty const& get(dimensions<Ty> const& dim) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return dim.width_;
  }
  else {
    return dim.height_;
  }
}

template<std::size_t Idx, std::unsigned_integral Ty>
Ty& get(dimensions<Ty>& dim) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return dim.width_;
  }
  else {
    return dim.height_;
  }
}

template<std::size_t Idx, std::unsigned_integral Ty>
Ty&& get(dimensions<Ty>&& dim) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return std::move(dim.width_);
  }
  else {
    return std::move(dim.height_);
  }
}

using offset_t = point<std::int32_t>;

struct offset_hash {
  [[nodiscard]] std::size_t operator()(offset_t const& off) const noexcept {
    std::size_t hashed = 2166136261U;

    hashed ^= static_cast<std::size_t>(off.x_);
    hashed *= 16777619U;

    hashed ^= static_cast<std::size_t>(off.y_);
    hashed *= 16777619U;

    return hashed;
  }
};

template<std::integral Ty, std::unsigned_integral Tx>
[[nodiscard]] inline std::ptrdiff_t
    to_index(point<Ty> offset, dimensions<Tx> const& dim) noexcept {
  return static_cast<std::ptrdiff_t>(dim.width_) * offset.y_ + offset.x_;
}

template<std::integral Ty>
struct limits {
  using value_type = Ty;

  inline void update(value_type value) noexcept {
    if (value > upper_) {
      upper_ = value;
    }
    else if (value < lower_) {
      lower_ = value;
    }
  }

  [[nodiscard]] inline value_type size() const noexcept {
    return upper_ - lower_;
  }

  value_type lower_{std::numeric_limits<value_type>::max()};
  value_type upper_{std::numeric_limits<value_type>::min()};
};

template<std::size_t Idx, std::integral Ty>
Ty const& get(limits<Ty> const& lim) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return lim.lower_;
  }
  else {
    return lim.upper_;
  }
}

template<std::size_t Idx, std::integral Ty>
Ty& get(limits<Ty>& lim) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return lim.lower_;
  }
  else {
    return lim.upper_;
  }
}

template<std::size_t Idx, std::integral Ty>
Ty&& get(limits<Ty>&& lim) {
  static_assert(Idx < 2);

  if constexpr (Idx == 0) {
    return std::move(lim.lower_);
  }
  else {
    return std::move(lim.upper_);
  }
}

template<std::integral Ty>
struct region {
  using value_type = Ty;

  [[nodicard]] inline point<value_type> left_top() const noexcept {
    return {left_, top_};
  }

  [[nodicard]] inline point<value_type> right_bottom() const noexcept {
    return {right_, bottom_};
  }

  [[nodiscard]] inline value_type width() const noexcept {
    return right_ - left_;
  }

  [[nodiscard]] inline value_type height() const noexcept {
    return bottom_ - top_;
  }

  [[nodiscard]] inline point<value_type> margins() const noexcept {
    return {left_ + right_, top_ + bottom_};
  }

  [[nodiscard]] inline value_type area() const noexcept {
    return width() * height();
  }

  [[nodiscard]] inline bool
      contains(point<value_type> const& point) const noexcept {
    return point.x_ >= left_ && point.x_ <= right_ && point.y_ >= top_ &&
           point.y_ <= bottom_;
  }

  value_type left_{};
  value_type top_{};
  value_type right_{};
  value_type bottom_{};
};

template<std::integral Ty>
[[nodiscard]] inline region<Ty> from_limits(limits<Ty> hor,
                                            limits<Ty> ver) noexcept {
  return {hor.lower_, ver.lower_, hor.upper_, ver.upper_};
}

template<std::size_t Idx, std::integral Ty>
Ty const& get(region<Ty> const& reg) {
  static_assert(Idx < 4);

  if constexpr (Idx == 0) {
    return reg.left_;
  }
  if constexpr (Idx == 1) {
    return reg.top_;
  }
  if constexpr (Idx == 2) {
    return reg.right_;
  }
  else {
    return reg.bottom_;
  }
}

template<std::size_t Idx, std::integral Ty>
Ty& get(region<Ty>& reg) {
  static_assert(Idx < 4);

  if constexpr (Idx == 0) {
    return reg.left_;
  }
  if constexpr (Idx == 1) {
    return reg.top_;
  }
  if constexpr (Idx == 2) {
    return reg.right_;
  }
  else {
    return reg.bottom_;
  }
}

template<std::size_t Idx, std::integral Ty>
Ty&& get(region<Ty>&& reg) {
  static_assert(Idx < 4);

  if constexpr (Idx == 0) {
    return std::move(reg.left_);
  }
  if constexpr (Idx == 1) {
    return std::move(reg.top_);
  }
  if constexpr (Idx == 2) {
    return std::move(reg.right_);
  }
  else {
    return std::move(reg.bottom_);
  }
}

} // namespace cdt
