
// action-window scanner

#pragma once

#include "cpl.hpp"
#include "cte.hpp"
#include "ifd.hpp"
#include "mrl.hpp"

#include <intrin.h>

namespace aws {
namespace details {
  using heatmap_t = mrl::matrix<cpl::mon_bv>;
  using contour_t = ctr::contour<cpl::mon_bv>;

  template<typename Image>
  inline constexpr auto pixel_size_v{sizeof(typename Image::value_type)};

  template<typename Mm, typename Image>
  inline constexpr auto step_size_v{sizeof(Mm) / pixel_size_v<Image>};

  template<typename Mm, typename Image>
  inline typename Image::value_type const*
      adjust_end(typename Image::value_type const* end) noexcept {
    return end - reinterpret_cast<std::uintptr_t>(end) % (sizeof(Mm) / 8) /
                     step_size_v<Mm, Image>;
  }

  template<typename Image>
  void compare(Image const& previous,
               Image const& current,
               heatmap_t& output) noexcept {
    using mm_t = __m256i;
    constexpr auto step{step_size_v<mm_t, Image>};

    auto o{output.data()};
    auto p{previous.data()}, c{current.data()};

    for (auto e{adjust_end<mm_t, Image>(current.end())}; c < e;
         p += step, c += step, o += step) {
      *reinterpret_cast<mm_t*>(o) = _mm256_and_si256(
          *reinterpret_cast<mm_t const*>(o),
          _mm256_cmpeq_epi8(*reinterpret_cast<mm_t const*>(p),
                            *reinterpret_cast<mm_t const*>(c)));
    }

    for (auto e{current.end()}; c < e; ++p, ++c, ++o) {
      if (*p != *c) {
        *o = {0};
      }
    }
  }

  [[nodiscard]] inline contour_t
      get_best(std::vector<contour_t>&& contours) noexcept {
    return *std::min_element(
        contours.begin(), contours.end(), [](auto& lhs, auto& rhs) {
          return lhs.area() * value(lhs.color()) <
                 rhs.area() * value(rhs.color());
        });
  }
} // namespace details

template<typename Feeder>
[[nodiscard]] std::optional<mrl::region_t>
    scan(Feeder&& feed, mrl::dimensions_t const& dimensions) requires(
        ifd::feeder<std::decay_t<Feeder>>) {
  cte::extractor<cpl::mon_bv> extractor{dimensions};
  details::heatmap_t heatmap{dimensions, {1}};

  auto mask{[](auto px, auto idx) { return value(px) != 0xff; }};

  std::optional<mrl::region_t> result{};
  if (feed.has_more()) {
    auto const min_area{dimensions.area() / 3};
    auto const min_height{2 * dimensions.height_ / 5};
    auto const min_width{2 * dimensions.width_ / 3};

    auto pimage{feed.produce()};
    for (std::size_t area{}, stagnation{};
         feed.has_more() && stagnation <= 100;) {
      auto cimage{feed.produce()};

      details::compare(pimage, cimage, heatmap);
      if (auto contour{details::get_best(extractor.extract(heatmap, mask))};
          value(contour.color()) == 0) {
        if (contour.area() > area) {
          stagnation = 0;
          area = contour.area();

          if (auto window{contour.enclosure()};
              result || area > min_area && window.height() > min_height &&
                            window.width() > min_width) {
            result = window;
          }
        }
      }

      if (result) {
        ++stagnation;
      }

      pimage = std::move(cimage);
    }
  }

  return mrl::region_t{
      result->left_ + 1,
      result->top_ + 1,
      result->right_ - 1,
      result->bottom_ - 1,
  };
}
} // namespace aws
