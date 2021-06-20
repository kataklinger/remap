
// foreground filtering

#pragma once

#include "fde.hpp"
#include "fgm.hpp"

#include <execution>
#include <iterator>

namespace fdf {

struct background {
  fgm::point_t zero_;
  mrl::matrix<cpl::nat_cc> image_;
};

namespace details {

  struct blend_item {
    fgm::frame frame_;
    background const* background_;
  };

  template<std::uint8_t Depth>
  class blender {
  public:
    using fragment_t = typename fgm::fragment<Depth>;

  private:
    class output {
    public:
      output(background const* bgd, mrl::dimensions_t const& dim)
          : background_{bgd}
          , extractor_{background_->image_, dim}
          , result_{background_->image_.dimensions(), background_->zero_} {
      }

      void update(mrl::matrix<cpl::nat_cc> const& image,
                  fgm::frame const& frame) {
        auto foreground{
            extractor_.extract(image, frame.position_ - result_.zero())};
        auto mask{fde::mask(foreground, image.dimensions())};

        result_.blit(frame.position_, image, mask, frame.number_);
      }

      [[nodiscard]] inline fragment_t result() noexcept {
        return std::move(result_);
      }

    private:
      background const* background_;

      fde::extractor<std::allocator<char>> extractor_;

      fragment_t result_;
    };

  public:
    inline void update(blend_item const& item,
                       mrl::matrix<cpl::nat_cc> const& image) {
      get_output(item.background_, image.dimensions())
          .update(image, item.frame_);
    }

    [[nodiscard]] std::vector<fragment_t> result() && noexcept {
      std::vector<fragment_t> ret{};
      for (auto& [k, r] : outputs_) {
        ret.emplace_back(r.result());
      }

      return ret;
    }

  private:
    [[nodiscard]] inline output&
        get_output(background const* bgd,
                   mrl::dimensions_t const& dim) noexcept {
      auto [it, success] = outputs_.try_emplace(bgd, bgd, dim);
      return it->second;
    }

  private:
    std::unordered_map<background const*, output> outputs_;
  };

  template<std::uint8_t Depth>
  [[nodiscard]] std::vector<background>
      get_background(std::vector<fgm::fragment<Depth>> const& fragments) {
    std::vector<background> results{fragments.size()};
    std::transform(std::execution::par,
                   fragments.begin(),
                   fragments.end(),
                   results.begin(),
                   [](auto& f) {
                     auto bkg{f.blend()};
                     return background{f.zero(), std::move(bkg.image_)};
                   });

    return results;
  }

  template<std::uint8_t Depth>
  [[nodiscard]] std::vector<blend_item>
      get_items(std::vector<fgm::fragment<Depth>> const& fragments,
                std::vector<background> const& backgrounds) {
    std::vector<blend_item> results{};

    std::size_t i{0};
    for (auto& fragment : fragments) {
      auto& frames{fragment.frames()};
      std::transform(frames.begin(),
                     frames.end(),
                     std::back_inserter(results),
                     [&backgrounds, i](auto& frame) {
                       return blend_item{frame, &backgrounds[i]};
                     });

      ++i;
    }

    std::sort(results.begin(), results.end(), [](auto& left, auto& right) {
      return left.frame_.number_ < right.frame_.number_;
    });

    return results;
  }

} // namespace details

template<std::uint8_t Depth, typename Feeder>
[[nodiscard]] std::vector<fgm::fragment<Depth>>
    filter(std::vector<fgm::fragment<Depth>> const& fragments,
           std::vector<background> const& backgrounds,
           Feeder&& feed) requires(ifd::feeder<std::decay_t<Feeder>>) {
  auto items{details::get_items(fragments, backgrounds)};

  details::blender<Depth> current{};

  std::size_t i{0};
  while (feed.has_more()) {
    auto [no, image]{feed.produce()};
    if (auto& item{items[i]}; no == item.frame_.number_) {
      current.update(item, image);
      ++i;
    }
  }

  return std::move(current).result();
}

template<std::uint8_t Depth, typename Feeder>
[[nodiscard]] inline std::vector<fgm::fragment<Depth>>
    filter(std::vector<fgm::fragment<Depth>> const& fragments,
           Feeder&& feed) requires(ifd::feeder<std::decay_t<Feeder>>) {
  return filter(fragments,
                details::get_background(fragments),
                std::forward<Feeder>(feed));
}

} // namespace fdf
