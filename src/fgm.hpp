
// fragmentation

#pragma once

#include "cpl.hpp"
#include "mrl.hpp"

#include <algorithm>

namespace fgm {
template<std::uint8_t Depth>
using dot_t = std::array<std::uint16_t, Depth>;

using point_t = cdt::point<std::int32_t>;

struct fragment_blend {
  mrl::matrix<cpl::nat_cc> image_;
  mrl::matrix<std::uint8_t> mask_;
};

struct frame {
  std::size_t number_;
  point_t position_;
};

struct no_content_tag {};
inline constexpr no_content_tag no_content{};

template<std::uint8_t Depth>
class fragment {
public:
  static inline constexpr auto depth{Depth};

  using dot_type = dot_t<depth>;
  using matrix_type = mrl::matrix<dot_type>;

public:
  inline fragment() noexcept
      : step_{1, 1}
      , dots_{step_} {
  }

  inline explicit fragment(mrl::dimensions_t step) noexcept
      : step_{step}
      , dots_{step} {
  }

  inline fragment(matrix_type dots,
                  mrl::dimensions_t step,
                  point_t zero,
                  std::vector<fgm::frame>&& frames) noexcept
      : step_{step}
      , dots_{std::move(dots)}
      , zero_{zero}
      , frames_{std::move(frames)} {
  }

  inline fragment(fragment const& other, no_content_tag /*unused*/) noexcept
      : step_{other.step_}
      , dots_{other.dots_.dimensions()}
      , zero_{other.zero_} {
  }

  template<typename Alloc1, typename Alloc2>
  void blit(point_t pos,
            mrl::matrix<cpl::nat_cc, Alloc1> const& image,
            mrl::matrix<cpl::mon_bv, Alloc2> const& mask) {
    blit_impl(pos, image, [m = mask.data()](auto dst, auto src) mutable {
      if (value(*(m++)) == 0) {
        ++(*dst)[value(*src)];
      }
    });
  }

  template<typename Alloc>
  void blit(point_t pos,
            mrl::matrix<cpl::nat_cc, Alloc> const& image,
            std::size_t frame_no) {
    ensure(pos, image.dimensions());

    blit_impl(pos, image, [](auto dst, auto src) { ++(*dst)[value(*src)]; });

    frames_.emplace_back(frame_no, pos);
  }

  void blit(point_t pos, fragment const& other) {
    ensure(pos, other.dots_.dimensions());

    blit_impl(pos, other.dots_, [](auto dst, auto src) {
      for (std::uint8_t i{0}; i < depth; ++i) {
        (*dst)[i] += (*src)[i];
      }
    });

    frames_.reserve(frames_.size() + other.frames_.size());
    for (auto& f : other.frames_) {
      frames_.emplace_back(f.number_, f.position_ - other.zero_ + pos);
    }
  }

  [[nodiscard]] fragment_blend blend() const {
    mrl::matrix<cpl::nat_cc> image{dots_.dimensions()};
    mrl::matrix<std::uint8_t> mask{dots_.dimensions()};

    auto img_out{image.data()};
    auto mask_out{mask.data()};

    for (auto first{dots_.data()}, last{dots_.end()}; first < last;
         ++first, ++img_out, ++mask_out) {
      auto dot{&(*first)[0]};
      auto selected{std::max_element(dot, dot + depth)};
      if (*selected != 0) {
        *img_out = {static_cast<cpl::nat_cc::value_type>(selected - dot)};
        *mask_out = *selected != 0 ? 1 : 0;
      }
    }

    return {std::move(image), std::move(mask)};
  }

  [[nodiscard]] inline matrix_type const& dots() const noexcept {
    return dots_;
  }

  [[nodiscard]] inline mrl::dimensions_t dimensions() const noexcept {
    return dots_.dimensions();
  }

  [[nodiscard]] inline mrl::dimensions_t step() const noexcept {
    return step_;
  }

  [[nodiscard]] inline point_t zero() const noexcept {
    return zero_;
  }

  [[nodiscard]] inline std::vector<frame> const& frames() const noexcept {
    return frames_;
  }

private:
  template<typename Ty, typename Alloc, typename Fn>
  void blit_impl(point_t pos, mrl::matrix<Ty, Alloc> const& source, Fn fn) {
    auto adj_x{pos.x_ - zero_.x_}, adj_y{pos.y_ - zero_.y_};
    auto stride{dots_.width() - source.width()};

    auto dst{dots_.data() + adj_x + adj_y * dots_.width()};
    for (auto first{source.data()}, last{source.end()}; first < last;
         dst += stride) {
      for (auto end{first + source.width()}; first < end; ++first, ++dst) {
        fn(dst, first);
      }
    }
  }

  void ensure(point_t pos, mrl::dimensions_t const& dim) {
    mrl::region_t region{};

    auto extend_h{extend<0>(region, pos, dim)};
    auto extend_v{extend<1>(region, pos, dim)};

    if (extend_h || extend_v) {
      dots_ = dots_.extend(region);
    }
  }

  template<std::size_t Idx>
  [[nodiscard]] bool extend(mrl::region_t& region,
                            point_t pos,
                            mrl::dimensions_t const& dim) noexcept {
    if (get<Idx>(pos) < get<Idx>(zero_)) {
      get<Idx>(region) = get_step<Idx>(get<Idx>(zero_) - get<Idx>(pos));
      get<Idx>(zero_) -= get<Idx>(region);

      return true;
    }

    auto required{get<Idx>(pos) + static_cast<std::int32_t>(get<Idx>(dim))};
    if (required > 0) {
      if (auto limit{get<Idx>(zero_) + get<Idx>(dots_.dimensions())};
          static_cast<std::size_t>(required) > limit) {
        get<Idx + 2>(region) = get_step<Idx>(required - limit);

        return true;
      }
    }

    return false;
  }

  template<std::size_t Idx>
  [[nodiscard]] inline std::size_t get_step(std::size_t change) const noexcept {
    auto step{get<Idx>(step_)};
    return (change / step) + (change % step != 0 ? step : 0);
  }

private:
  mrl::dimensions_t step_;
  matrix_type dots_;

  point_t zero_;

  std::vector<frame> frames_;
};
} // namespace fgm
