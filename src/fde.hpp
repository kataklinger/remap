
// foreground extraction

#pragma once

#include "cte.hpp"
#include "fgm.hpp"
#include "kpe.hpp"

#include <intrin.h>

namespace fde {

namespace details {

  using mm_type = __m256i;
  constexpr inline auto mm_size{sizeof(mm_type)};

  template<typename TAlign>
  void generate_mask(sid::nat::dimg_t const& background,
                     sid::nat::dimg_t const& frame,
                     sid::mon::dimg_t& output,
                     std::size_t idx,
                     TAlign /*unused*/) noexcept {
    using namespace cpl;

    auto &bdim{background.dimensions()}, &fdim{frame.dimensions()};
    auto vec_size{fdim.width_ - fdim.width_ % mm_size};

    auto out{output.data()};
    for (auto brow{background.data() + idx},
         fcur{frame.data()},
         fend{frame.end()};
         fcur < fend;
         brow += bdim.width_) {
      auto bcur{brow};

      for (auto bend{brow + vec_size}; bcur < bend;
           bcur += mm_size, fcur += mm_size, out += mm_size) {
        if constexpr (TAlign::value) {
          *reinterpret_cast<mm_type*>(out) =
              _mm256_cmpeq_epi8(*reinterpret_cast<mm_type const*>(bcur),
                                *reinterpret_cast<mm_type const*>(fcur));
        }
        else {
          *reinterpret_cast<mm_type*>(out) = _mm256_cmpeq_epi8(
              _mm256_loadu_epi8(bcur), *reinterpret_cast<mm_type const*>(fcur));
        }
      }

      for (auto bend{brow + fdim.width_}; bcur < bend; ++bcur, ++fcur, ++out) {
        *out = *bcur == *fcur ? 0xff_bv : 0_bv;
      }
    }
  }

  template<typename Alloc>
  using extractor_t =
      cte::extractor<cpl::nat_cc, all::rebind_alloc_t<Alloc, ctr::edge>>;

} // namespace details

template<typename Alloc>
using contours_t = typename details::extractor_t<Alloc>::contours;

template<typename Alloc>
class extractor {
public:
  using allocator_type = Alloc;

  using forground_t =
      std::vector<all::rebind_alloc_t<allocator_type, ctr::edge>>;

public:
  extractor(sid::nat::dimg_t const& background,
            mrl::dimensions_t dimensions,
            allocator_type const& allocator = allocator_type{})
      : background_{&background}
      , contours_{dimensions, allocator}
      , mask_{dimensions} {
  }

  [[nodiscard]] contours_t<allocator_type>
      extract(sid::nat::dimg_t const& frame,
              sid::nat::dimg_t const& median,
              fgm::point_t position) {
    generate_mask(frame, cdt::to_index(position, background_->dimensions()));

    auto forground{
        contours_.extract(median, [m = mask_.data()](auto px, auto idx) {
          return value(m[idx]) == 0;
        })};

    auto area_limit{frame.dimensions().area() / 5};

    forground.erase(
        std::remove_if(forground.begin(),
                       forground.end(),
                       [area_limit](auto& c) { return c.area() > area_limit; }),
        forground.end());

    return forground;
  }

private:
  void generate_mask(sid::nat::dimg_t const& frame, std::size_t idx) noexcept {
    if (idx % details::mm_size == 0) {
      details::generate_mask(*background_, frame, mask_, idx, std::true_type{});
    }
    else {
      details::generate_mask(
          *background_, frame, mask_, idx, std::false_type{});
    }
  }

private:
  details::extractor_t<allocator_type> contours_;
  sid::nat::dimg_t const* background_;
  sid::mon::dimg_t mask_;
};

template<typename Contour, typename Alloc>
[[nodiscard]] auto mask(std::vector<Contour, Alloc> const& forground,
                        mrl::dimensions_t const& dim) {
  using alloc_t = all::rebind_alloc_t<Alloc, cpl::mon_bv>;

  sid::mon::aimg_t<alloc_t> result{dim, forground.get_allocator()};

  auto output{result.data()};
  for (auto& contour : forground) {
    contour.recover(output, {1});
  }

  for (auto& contour : forground) {
    auto& reg = contour.enclosure();
    output = result.data() + reg.top_ * dim.width_;

    for (auto y{reg.top_}; y < reg.bottom_; ++y, output += dim.width_) {
      for (auto x{reg.left_}; x < reg.right_; ++x) {
        output[x] = {1};
      }
    }
  }

  return result;
}

} // namespace fde
