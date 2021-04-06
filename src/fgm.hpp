
#pragma once

#include "cpl.hpp"
#include "mrl.hpp"

#include <algorithm>

namespace fgm {
template<std::uint8_t Depth>
using dot_t = std::array<std::uint16_t, Depth>;

using point_t = cdt::point<std::int32_t>;

template<std::uint8_t Depth, cpl::pixel Pixel>
class fragment {
public:
  static inline constexpr auto depth{Depth};

  using pixel_type = Pixel;

  using dot_type = dot_t<depth>;
  using matrix_type = mrl::matrix<dot_type>;

public:
  inline fragment(mrl::dimensions_t step) noexcept
      : step_{step}
      , dots_{step} {
  }

  template<typename Alloc>
  void blit(point_t pos, mrl::matrix<pixel_type, Alloc> const& image) {
    ensure(pos);

    auto adj_x{pos.x_ - zero_.x_}, adj_y{pos.y_ - zero_.y_};
    auto stride{dots_.width() - image.width()};

    auto out{dots_.data() + adj_x + adj_y * dots_.width()};
    for (auto first{image.data()}, last{image.end()}; first < last;
         out += stride) {
      for (auto end{first + image.width()}; first < end; ++first, ++out) {
        ++(*out)[value(*first)];
      }
    }
  }

  void blit(point_t pos, fragment const& other) {
    ensure(pos);

    auto& dots{other.dots_};
    auto adj_x{pos.x_ - zero_.x_}, adj_y{pos.y_ - zero_.y_};
    auto stride{dots_.width() - dots.width()};

    auto out{dots_.data() + adj_x + adj_y * dots_.width()};
    for (auto first{dots.data()}, last{dots.end()}; first < last;
         out += stride) {
      for (auto end{first + dots.width()}; first < end; ++first, ++out) {
        for (std::uint8_t i{0}; i < depth; ++i) {
          (*out)[i] += (*first)[i];
        }
      }
    }
  }

  [[nodiscard]] mrl::matrix<pixel_type> generate() const {
    mrl::matrix<pixel_type> result{dots_.dimensions()};
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

  [[nodiscard]] inline mrl::dimensions_t dimensions() const noexcept {
    return dots_.dimensions();
  }

private:
  void ensure(point_t pos) {
    mrl::region_t region{};

    auto extend_h{extend<0>(region, pos, dots_.width())};
    auto extend_v{extend<1>(region, pos, dots_.height())};

    if (extend_h || extend_v) {
      dots_ = dots_.extend(region);
    }
  }

  template<std::size_t Idx>
  [[nodiscard]] bool
      extend(mrl::region_t& region, point_t pos, std::size_t size) noexcept {
    if (get<Idx>(pos) < get<Idx>(zero_)) {
      get<Idx>(region) = get_step<Idx>(pos);
      get<Idx>(zero_) -= get<Idx>(region);

      return true;
    }

    if (auto limit{get_limit<Idx>(size)}; get<Idx>(pos) > limit) {
      get<Idx + 2>(region) = get_step<Idx>(pos, limit);

      return true;
    }

    return false;
  }

  template<std::size_t Idx>
  [[nodiscard]] inline std::size_t get_step(point_t pos) const noexcept {
    return std::max(get<Idx>(step_),
                    static_cast<std::size_t>(get<Idx>(zero_) - get<Idx>(pos)));
  }

  template<std::size_t Idx>
  [[nodiscard]] inline std::size_t get_step(point_t pos,
                                            std::size_t limit) const noexcept {
    return std::max(get<Idx>(step_),
                    static_cast<std::size_t>(get<Idx>(pos) - limit));
  }

  template<std::size_t Idx>
  [[nodiscard]] inline std::int32_t get_limit(std::size_t size) const noexcept {
    return get<Idx>(zero_) + size - get<Idx>(step_);
  }

private:
  mrl::dimensions_t step_;
  matrix_type dots_;

  point_t zero_;
};
} // namespace fgm
