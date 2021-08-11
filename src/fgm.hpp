
// fragmentation

#pragma once

#include "icd.hpp"

#include <algorithm>

namespace fgm {

inline constexpr std::uint8_t depth{16};

using dot_type = std::array<std::uint16_t, depth>;

using point_t = cdt::point<std::int32_t>;

struct fragment_blend {
  sid::nat::dimg_t image_;
  sid::mon::dimg_t mask_;
};

struct packed_data {
  icd::compressed_t image_;
  icd::compressed_t median_;
};

struct frame {
  std::size_t number_;
  point_t position_;

  packed_data data_;
};

class fragment {
public:
  using matrix_type = mrl::matrix<dot_type>;

public:
  inline fragment() noexcept
      : step_{1, 1}
      , dots_{step_} {
  }

  inline explicit fragment(mrl::dimensions_t const& step) noexcept
      : step_{step}
      , dots_{step} {
  }

  inline fragment(matrix_type dots,
                  mrl::dimensions_t const& step,
                  point_t const& zero,
                  std::vector<fgm::frame>&& frames) noexcept
      : step_{step}
      , dots_{std::move(dots)}
      , zero_{zero}
      , frames_{std::move(frames)} {
  }

  inline fragment(mrl::dimensions_t const& dimensions,
                  point_t const& zero) noexcept
      : step_{1, 1}
      , dots_{dimensions}
      , zero_{zero} {
  }

  template<typename Alloc1, typename Alloc2>
  void blit(point_t pos,
            sid::nat::aimg_t<Alloc1> const& image,
            sid::mon::aimg_t<Alloc2> const& mask,
            std::size_t frame_no) {
    ensure(pos, image.dimensions());

    blit_impl(pos, image, [m = mask.data()](auto dst, auto src) mutable {
      if (value(*(m++)) == 0) {
        ++(*dst)[value(*src)];
      }
    });

    frames_.emplace_back(frame_no, pos);
  }

  template<typename Alloc>
  void blit(point_t pos,
            sid::nat::aimg_t<Alloc> const& image,
            packed_data&& packed,
            std::size_t frame_no) {
    ensure(pos, image.dimensions());

    blit_impl(pos, image, [](auto dst, auto src) { ++(*dst)[value(*src)]; });

    frames_.emplace_back(frame_no, pos, std::move(packed));
  }

  void blit(point_t pos, fragment&& other) {
    ensure(pos, other.dots_.dimensions());

    blit_impl(pos, other.dots_, [](auto dst, auto src) {
      for (std::uint8_t i{0}; i < depth; ++i) {
        (*dst)[i] += (*src)[i];
      }
    });

    frames_.reserve(frames_.size() + other.frames_.size());
    for (auto& f : other.frames_) {
      frames_.emplace_back(
          f.number_, f.position_ - other.zero_ + pos, std::move(f.data_));
    }
  }

  [[nodiscard]] fragment_blend blend() const {
    using namespace cpl;

    sid::nat::dimg_t image{dots_.dimensions()};
    sid::mon::dimg_t mask{dots_.dimensions()};

    auto img_out{image.data()};
    auto mask_out{mask.data()};

    for (auto first{dots_.data()}, last{dots_.end()}; first < last;
         ++first, ++img_out, ++mask_out) {
      auto dot{&(*first)[0]};
      auto selected{std::max_element(dot, dot + depth)};
      if (*selected != 0) {
        *img_out = {static_cast<cpl::nat_cc::value_type>(selected - dot)};
        *mask_out = *selected != 0 ? 1_bv : 0_bv;
      }
    }

    return {std::move(image), std::move(mask)};
  }

  void normalize() noexcept {
    for (auto& frame : frames_) {
      frame.position_ -= zero_;
    }

    zero_ = {0, 0};
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
    auto extended{false};

    if (get<Idx>(pos) < get<Idx>(zero_)) {
      get<Idx>(region) = get_step<Idx>(get<Idx>(zero_) - get<Idx>(pos));

      extended = true;
    }

    auto required{get<Idx>(pos) + static_cast<std::int32_t>(get<Idx>(dim))};
    if (required > 0) {
      if (auto limit{get<Idx>(zero_) + get<Idx>(dots_.dimensions())};
          static_cast<std::size_t>(required) > limit) {
        get<Idx + 2>(region) = get_step<Idx>(required - limit);

        extended = true;
      }
    }

    get<Idx>(zero_) -= get<Idx>(region);

    return extended;
  }

  template<std::size_t Idx>
  [[nodiscard]] inline std::size_t get_step(std::size_t change) const noexcept {
    auto step{get<Idx>(step_)};
    auto rest{change % step};
    return (change - rest) + (rest != 0 ? step : 0);
  }

private:
  mrl::dimensions_t step_;
  matrix_type dots_;

  point_t zero_;

  std::vector<frame> frames_;
};

} // namespace fgm
