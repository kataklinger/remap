
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
};

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

template<std::unsigned_integral Ty>
struct dimensions {
  using value_type = Ty;

  [[nodicard]] inline value_type area() const noexcept {
    return width_ * height_;
  }

  value_type width_{};
  value_type height_{};
};

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

template<std::integral Ty>
struct region {
  using value_type = Ty;

  [[nodiscard]] inline value_type width() const noexcept {
    return right_ - left_;
  }

  [[nodiscard]] inline value_type height() const noexcept {
    return bottom_ - top_;
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

} // namespace cdt
