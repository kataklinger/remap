
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

  template<std::uint8_t Depth>
  [[nodiscard]] std::vector<background>
      get_background(std::vector<fgm::fragment<Depth>> const& fragments) {
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

template<typename Comp, std::uint8_t Depth>
[[nodiscard]] std::vector<fgm::fragment<Depth>> filter(
    std::vector<fgm::fragment<Depth>> const& fragments,
    std::vector<background> const& backgrounds,
    mrl::dimensions_t const& frame_dim,
    Comp&& comp) requires(icd::decompressor<std::decay_t<Comp>,
                                            std::allocator<cpl::nat_cc>>) {
  std::vector<fgm::fragment<Depth>> results{};

  for (std::size_t i{0}, l{fragments.size()}; i < l; ++i) {
    auto& fragment = fragments[i];
    auto& background = backgrounds[i];

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
    }
  }

  return results;
}

template<typename Comp, std::uint8_t Depth>
[[nodiscard]] inline std::vector<fgm::fragment<Depth>>
    filter(std::vector<fgm::fragment<Depth>> const& fragments,
           mrl::dimensions_t const& frame_dim,
           Comp&& comp) {
  return filter(fragments,
                details::get_background(fragments),
                frame_dim,
                std::forward<Comp>(comp));
}

} // namespace fdf
