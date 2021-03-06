
#pragma once

#include "cpl.hpp"
#include "mrl.hpp"

namespace fgm {
template<std::uint8_t Depth>
using dot_t = std::array<std::uint16_t, Depth>;

template<std::uint8_t Depth, cpl::pixel Pixel>
class fragment {
public:
  static inline constexpr auto depth{Depth};

  using pixel_type = Pixel;

  using dot_type = dot_t<depth>;
  using matrix_type = mrl::matrix<dot_type>;
  using step_type = mrl::size_type;

public:
  inline fragment(step_type hstep, step_type vstep) noexcept
      : hstep_{hstep}
      , vstep_{vstep}
      , dots_{hstep, vstep} {
  }

  template<typename Alloc>
  void blit(std::int32_t x,
            std::int32_t y,
            mrl::matrix<pixel_type, Alloc> const& image) {
    ensure(x, y);

    auto adj_x{x - x_}, adj_y{y - y_};
    auto stride{dots_.width() - image.width()};

    auto out{dots_.data() + adj_x + adj_y * dots_.width()};
    for (auto first{image.data()}, last{image.end()}; first < last;
         out += stride) {
      for (auto end{first + image.width()}; first < end; ++first, ++out) {
        ++(*out)[value(*first)];
      }
    }
  }

  [[nodiscard]] mrl::matrix<pixel_type> generate() const {
    mrl::matrix<pixel_type> result{dots_.width(), dots_.height()};
    auto out{result.data()};
    for (auto first{dots_.data()}, last{dots_.end()}; first < last;
         ++first, ++out) {
      auto dot{&(*first)[0]};
      auto selected{std::max_element(dot, dot + depth)};
      if (*selected != 0) {
        *out = {static_cast<typename pixel_type::value_type>(selected - dot)};
      }
    }

    return result;
  }

private:
  void ensure(std::int32_t x, std::int32_t y) {
    step_type left{0}, right{0}, top{0}, bottom{0};

    auto extend{false};
    if (x < x_) {
      left = hstep_;
      x_ -= hstep_;
      extend = true;
    }
    else if (x > static_cast<std::int32_t>(x_ + dots_.width() - hstep_)) {
      right = hstep_;
      extend = true;
    }

    if (y < y_) {
      top = vstep_;
      y_ -= vstep_;
      extend = true;
    }
    else if (y > static_cast<std::int32_t>(y_ + dots_.height() - vstep_)) {
      bottom = vstep_;
      extend = true;
    }

    if (extend) {
      dots_ = dots_.extend(left, right, top, bottom);
    }
  }

private:
  step_type hstep_;
  step_type vstep_;

  matrix_type dots_;

  std::int32_t x_{0};
  std::int32_t y_{0};
};
} // namespace fgm
