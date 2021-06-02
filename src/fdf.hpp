
// foreground filtering

#pragma once

#include "fde.hpp"
#include "fgm.hpp"

#include <iterator>

namespace fdf {

namespace details {
  template<std::uint8_t Depth>
  struct blend_item {
    using fragment_t = fgm::fragment<Depth>;

    fgm::frame frame_;
    fragment_t const* original_;
  };

  template<std::uint8_t Depth>
  class blender {
  public:
    using blend_t = blend_item<Depth>;
    using fragment_t = typename blend_t::fragment_t;

  public:
    void update(blend_t& item, mrl::matrix<cpl::nat_cc>& image) {
      if (!extractor_.has_value() || original_ != item.original_) {
        original_ = item.original_;

        background_ = original_->blend();
        extractor_.emplace(background_.image_, image.dimensions());

        output_.emplace_back(*original_, fgm::no_content);
      }

      auto foreground{extractor_->extract(image, item.frame_.position_)};
      auto mask{fde::mask(foreground, image.dimensions())};

      output_.back().blit(item.frame_.position_, image, mask);
    }

    [[nodiscard]] std::vector<fragment_t> result() && noexcept {
      return std::move(output_);
    }

  private:
    fgm::fragment_blend background_;
    std::optional<fde::extractor<std::allocator<char>>> extractor_;

    fragment_t const* original_;
    std::vector<fragment_t> output_;
  };

} // namespace details

template<std::uint8_t Depth, typename Feeder>
[[nodiscard]] std::vector<fgm::fragment<Depth>>
    filter(std::vector<fgm::fragment<Depth>> const& fragments,
           Feeder&& feed) requires(ifd::feeder<std::decay_t<Feeder>>) {
  std::vector<details::blend_item<Depth>> blends{};
  for (auto& fragment : fragments) {
    auto& frames{fragment.frames()};
    std::transform(frames.begin(),
                   frames.end(),
                   std::back_inserter(blends),
                   [&fragment](auto& frame) {
                     return details::blend_item<Depth>{frame, &fragment};
                   });
  }

  std::sort(blends.begin(), blends.end(), [](auto& left, auto& right) {
    return left.frame_.number_ < right.frame_.number_;
  });

  details::blender<Depth> current{};

  while (feed.has_more()) {
    auto [no, image]{feed.produce()};
    if (auto& blend{blends[no]}; no == blend.frame_.number_) {
      current.update(blend, image);
    }
  }

  return std::move(current).result();
}
} // namespace fdf
