
// foreground filtering

#pragma once

#include "fde.hpp"
#include "fgm.hpp"

#include <execution>
#include <iterator>

namespace fdf {

struct background {
  fgm::point_t zero_;
  sid::nat::dimg_t image_;
};

namespace details {

  [[nodiscard]] inline std::vector<background>
      get_background(std::list<fgm::fragment> const& fragments) {
    std::vector<background> results{fragments.size()};
    std::transform(std::execution::par,
                   fragments.begin(),
                   fragments.end(),
                   results.begin(),
                   [](auto& frag) {
                     auto bkg{frag.blend()};
                     return background{frag.zero(), std::move(bkg.image_)};
                   });

    return results;
  }

} // namespace details

using contours_t = fde::contours_t<std::allocator<cpl::nat_cc>>;

template<typename Comp, typename Callback>
[[nodiscard]] std::vector<fgm::fragment> filter(
    std::list<fgm::fragment> const& fragments,
    std::vector<background> const& backgrounds,
    mrl::dimensions_t const& frame_dim,
    Comp&& comp,
    Callback&& cb) requires(icd::decompressor<std::decay_t<Comp>,
                                              std::allocator<cpl::nat_cc>>) {
  std::vector<fgm::fragment> results{};

  std::size_t i{0};
  for (auto& fragment : fragments) {
    auto& background{backgrounds[i]};

    fde::extractor<std::allocator<char>> extractor{background.image_,
                                                   frame_dim};

    auto& result{
        results.emplace_back(background.image_.dimensions(), background.zero_)};

    for (auto& [no, pos, data] : fragment.frames()) {
      auto image{comp(data.image_, frame_dim)};
      auto median{comp(data.median_, frame_dim)};

      auto foreground{extractor.extract(image, median, pos - result.zero())};
      auto mask{fde::mask(foreground, image.dimensions())};
      result.blit(pos, image, mask, no);

      cb(result, i, image, no, median, pos, foreground, mask);
    }

    ++i;
  }

  return results;
}

template<typename Comp, typename Callback>
[[nodiscard]] inline std::vector<fgm::fragment> filter(
    std::list<fgm::fragment> const& fragments,
    mrl::dimensions_t const& frame_dim,
    Comp&& comp,
    Callback&& cb) requires(icd::decompressor<std::decay_t<Comp>,
                                              std::allocator<cpl::nat_cc>>) {
  return filter(fragments,
                details::get_background(fragments),
                frame_dim,
                std::forward<Comp>(comp),
                std::forward<Callback>(cb));
}

} // namespace fdf
