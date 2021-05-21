
// foreground extraction

#pragma once

#include "cpl.hpp"
#include "cte.hpp"
#include "mrl.hpp"

#include <intrin.h>

namespace fde {
namespace details {
  using mm_type = __m256i;
  constexpr inline auto mm_size{sizeof(mm_type)};

  template<typename TAlign>
  void generate_mask(mrl::matrix<cpl::nat_cc> const& background,
                     mrl::matrix<cpl::nat_cc> const& frame,
                     mrl::matrix<std::uint8_t>& output,
                     std::size_t idx,
                     TAlign /*unused*/) noexcept {
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
        *out = *bcur == *fcur ? 0xff : 0;
      }
    }
  }
} // namespace details

template<typename Alloc>
class extractor {
public:
  using allocator_type = Alloc;

  using forground_t =
      std::vector<all::rebind_alloc_t<allocator_type, ctr::edge>>;

private:
  using contour_extractor_t =
      cte::extractor<cpl::nat_cc,
                     all::rebind_alloc_t<allocator_type, ctr::edge>>;

public:
  using contours = typename contour_extractor_t::contours;

public:
  extractor(mrl::matrix<cpl::nat_cc> const& background,
            mrl::dimensions_t dimensions,
            allocator_type const& allocator = allocator_type{})
      : background_{&background}
      , contours_{dimensions, allocator}
      , mask_{dimensions} {
  }

  [[nodiscard]] contours extract(mrl::matrix<cpl::nat_cc> const& frame,
                                 mrl::point_t position) {
    generate_mask(frame, cdt::to_index(position, background_->dimensions()));

    auto area_limit{frame.dimensions().area() / 5};

    auto forground{contours_.extract(
        frame, [m = mask_.data()](auto px, auto idx) { return m[idx] == 0; })};

    forground.erase(
        std::remove_if(forground.begin(),
                       forground.end(),
                       [area_limit](auto& c) { return c.area() > area_limit; }),
        forground.end());

    return forground;
  }

private:
  void generate_mask(mrl::matrix<cpl::nat_cc> const& frame,
                     std::size_t idx) noexcept {
    if (idx % details::mm_size == 0) {
      details::generate_mask(*background_, frame, mask_, idx, std::true_type{});
    }
    else {
      details::generate_mask(
          *background_, frame, mask_, idx, std::false_type{});
    }
  }

private:
  contour_extractor_t contours_;
  mrl::matrix<cpl::nat_cc> const* background_;

public:
  mrl::matrix<std::uint8_t> mask_;
};

template<typename Contour, typename Alloc>
[[nodiscard]] auto mask(std::vector<Contour, Alloc> const& forground,
                        mrl::dimensions_t const& dim) {
  using alloc_t = all::rebind_alloc_t<Alloc, cpl::mon_bv>;

  mrl::matrix<cpl::mon_bv, alloc_t> result{dim, forground.get_allocator()};

  auto output{result.data()};
  for (auto& contour : forground) {
    contour.recover(output, {1});
  }

  return result;
}

} // namespace fde
