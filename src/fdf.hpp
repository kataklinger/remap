
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

  private:
    class output {
    public:
      output(fragment_t const* original, mrl::dimensions_t const& dim)
          : zero_{original->zero()}
          , background_{original->blend()}
          , extractor_{background_.image_, dim}
          , result_{*original, fgm::no_content} {
      }

      void update(mrl::matrix<cpl::nat_cc> const& image,
                  fgm::point_t const& pos) {
        auto foreground{extractor_.extract(image, pos - zero_)};
        auto mask{fde::mask(foreground, image.dimensions())};

        result_.blit(pos, image, mask);
      }

      [[nodiscard]] inline fragment_t result() noexcept {
        return std::move(result_);
      }

    private:
      fgm::point_t zero_;
      fgm::fragment_blend background_;

      fde::extractor<std::allocator<char>> extractor_;

      fragment_t result_;
    };

  public:
    inline void update(blend_t const& item,
                       mrl::matrix<cpl::nat_cc> const& image) {
      get_output(item.original_, image.dimensions())
          .update(image, item.frame_.position_);
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
        get_output(fragment_t const* original,
                   mrl::dimensions_t const& dim) noexcept {
      auto [it, success] = outputs_.try_emplace(original, original, dim);
      return it->second;
    }

  private:
    std::unordered_map<fragment_t const*, output> outputs_;
  };

} // namespace details

template<std::uint8_t Depth, typename Feeder>
[[nodiscard]] std::vector<fgm::fragment<Depth>>
    filter_all(std::vector<fgm::fragment<Depth>> const& fragments,
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
    if (auto& blend{blends[no - 1]}; no == blend.frame_.number_) {
      current.update(blend, image);
    }
  }

  return std::move(current).result();
}
} // namespace fdf
