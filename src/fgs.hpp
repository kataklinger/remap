
#pragma once

#include "fgm.hpp"

namespace fgs {
namespace details {
  using grid_t = kpr::grid<1, 1, std::allocator<char>>;

  template<std::uint8_t Depth>
  using snippet_t = std::pair<fgm::fragment<Depth, cpl::nat_cc> const*, grid_t>;

  template<std::uint8_t Depth>
  [[nodiscard]] snippet_t<Depth>
      extract_single(fgm::fragment<Depth, cpl::nat_cc> const& fragment) {
    auto image{fragment.generate()};

    mrl::matrix<cpl::nat_cc> median{
        image.width(), image.height(), image.get_allocator()};

    kpe::extractor<grid_t, 0> extractor{image.width(), image.height()};
    return {&fragment, extractor.extract(image, median)};
  }

  template<std::uint8_t Depth, typename Iter>
  [[nodiscard]] auto extract_all(Iter first, Iter last) {
    using snip_t = details::snippet_t<Depth>;
    std::vector<snip_t> snippets(std::distance(first, last), snip_t{});

    std::transform(
        std::execution::par, first, last, snippets.begin(), [](auto& fragment) {
          return extract_single(fragment);
        });

    return snippets;
  }

  struct match_config {
    using allocator_type = std::allocator<char>;
    static constexpr std::size_t weight_switch{100};
    static constexpr std::size_t region_votes{3};

    [[nodiscard]] inline allocator_type get_allocator() const noexcept {
      return allocator_;
    }

    [[no_unique_address]] allocator_type allocator_;
  };

  using couple_t = std::pair<std::int16_t, std::int16_t>;

  class delta {
  public:
    using total_t = kpm::totalizator_t<match_config>;

  public:
    inline explicit delta(couple_t edge) noexcept
        : edge_{edge} {
    }

    inline void update_raw(kpm::ticket_t<match_config> const& ticket) noexcept {
      for (auto& [offset, count] : ticket) {
        matches_[offset] += count;
        max_count_ = std::max(max_count_, count);
      }
    }

    inline void crossmatch(cdt::offset_t const& offset) noexcept {
      matches_[offset] += max_count_;
    }

    [[nodiscard]] inline couple_t edge() const noexcept {
      return edge_;
    }

    [[nodiscard]] inline total_t const& raw() const noexcept {
      return matches_;
    }

    [[nodiscard]] inline cdt::offset_t best() const noexcept {
      return std::max_element(
                 matches_.begin(),
                 matches_.end(),
                 [](auto& lhs, auto& rhs) { return lhs.second < rhs.second; })
          ->first;
    }

  private:
    couple_t edge_;

    total_t matches_;
    std::size_t max_count_{0};
  };

  template<std::uint8_t Depth>
  [[nodiscard]] auto
      build_deltas(std::vector<snippet_t<Depth>> const& snippets) {
    std::vector<details::delta> deltas;

    auto segments{static_cast<std::int16_t>(snippets.size())};
    deltas.reserve((segments * segments - segments) / 2);

    for (int16_t i{segments - 1}; i >= 0; --i) {
      for (int16_t j{segments - 1}; j > i; --j) {
        deltas.emplace_back(couple_t{i, j});
      }
    }

    return deltas;
  }

  template<std::uint8_t Depth>
  void match_all(std::vector<snippet_t<Depth>> const& snippets,
                 std::vector<delta>& deltas) {
    std::for_each(std::execution::par,
                  deltas.begin(),
                  deltas.end(),
                  [&snippets](auto& d) {
                    auto [left, right]{d.edge()};
                    auto ticket{kpm::vote(details::match_config{},
                                          std::get<1>(snippets[left])[0],
                                          std::get<1>(snippets[right])[0])};
                    d.update_raw(ticket);
                  });
  }

  void crossmatch_single(delta& pi, delta& pj, delta& pk) noexcept {
    for (auto& [i, u1] : pi.raw()) {
      auto [ix, iy]{i};
      for (auto& [j, u2] : pj.raw()) {
        auto [jx, jy]{j};
        for (auto& [k, u3] : pk.raw()) {
          auto [kx, ky]{k};
          if (ix == jx + kx && iy == jy + ky) {
            pi.crossmatch(i);
            pj.crossmatch(j);
            pk.crossmatch(k);
          }
        }
      }
    }
  }

  void crossmatch_all(std::vector<delta>& deltas,
                      std::size_t segments) noexcept {
    for (auto i{deltas.begin()}; i < deltas.end(); ++i) {
      auto [ix, iy]{i->edge()};

      for (auto j{i + 1}; true; ++j) {
        if (auto [jx, jy]{j->edge()}; jx == ix) {
          auto k{deltas[jx * (2 * segments - jx - 1) / 2 + jy - 1]};

          crossmatch_single(*i, *j, k);
        }
        else {
          break;
        }
      }
    }
  }
} // namespace details

template<std::uint8_t Depth, typename Iter>
[[nodiscard]] mrl::matrix<cpl::nat_cc> splice(Iter first, Iter last) {
  auto snippets{details::extract_all<Depth>(first, last)};
  auto deltas{details::build_deltas(snippets)};
  details::match_all(snippets, deltas);
  details::crossmatch_all(deltas, snippets.size());

  return {};
}
} // namespace fgs
