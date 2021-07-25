
// fragment splicing

#pragma once

#include "fgm.hpp"
#include "kpe.hpp"
#include "kpm.hpp"

#include <execution>
#include <stack>

namespace fgs {
namespace details {
  using grid_t = kpr::grid<1, 1, std::allocator<char>>;

  template<std::uint8_t Depth>
  struct snippet {
    fgm::fragment<Depth> fragment_{};
    sid::mon::dimg_t mask_;

    grid_t grid_;
  };

  template<std::uint8_t Depth>
  [[nodiscard]] snippet<Depth> extract_single(fgm::fragment<Depth>&& fragment) {
    auto [image, mask]{fragment.blend()};

    sid::nat::dimg_t median{image.dimensions(), image.get_allocator()};

    kpe::extractor<grid_t, 0> extractor{image.dimensions()};
    return {std::move(fragment),
            std::move(mask),
            extractor.extract(image, median, image.get_allocator())};
  }

  template<std::uint8_t Depth, typename Iter>
  [[nodiscard]] auto extract_all(Iter first, Iter last) {
    using snippet_type = details::snippet<Depth>;
    std::vector<snippet_type> snippets(std::distance(first, last),
                                       snippet_type{});

    std::transform(
        std::execution::par, first, last, snippets.begin(), [](auto& fragment) {
          return extract_single(std::move(fragment));
        });

    return snippets;
  }

  struct match_config {
    using allocator_type = std::allocator<char>;

    static constexpr std::size_t weight_switch{
        std::numeric_limits<std::size_t>::max()};
    static constexpr std::size_t region_votes{3};

    [[nodiscard]] inline allocator_type get_allocator() const noexcept {
      return allocator_;
    }

    [[no_unique_address]] allocator_type allocator_;
  };

  using couple_t = std::pair<std::int16_t, std::int16_t>;

  class delta {
  public:
    struct match_value {
      std::uint8_t cross_;
      std::size_t keypoints_;

      friend auto operator<=>(match_value const&, match_value const&) = default;

      inline explicit match_value(std::size_t raw) noexcept
          : cross_{raw & 3}
          , keypoints_{raw >> 2} {
      }
    };

    struct match {
      cdt::offset_t offset_;
      match_value value_;

      inline match(
          std::pair<cdt::offset_t const, std::size_t> const& pair) noexcept
          : offset_{std::get<0>(pair)}
          , value_(std::get<1>(pair)) {
      }
    };

    using total_t = kpm::totalizator_t<match_config>;

  public:
    inline explicit delta(couple_t couple) noexcept
        : couple_{couple} {
    }

    inline void update_raw(kpm::ticket_t<match_config> const& ticket) noexcept {
      for (auto& [offset, count] : ticket) {
        matches_[offset] += (count << 2);
      }
    }

    inline void crossmatch(cdt::offset_t const& offset) noexcept {
      ++matches_[offset];
    }

    [[nodiscard]] inline couple_t couple() const noexcept {
      return couple_;
    }

    template<typename Ty>
    [[nodiscard]] inline auto
        couple(std::vector<Ty> const& data) const noexcept {
      auto [left, right]{couple_};
      return std::tuple<Ty const&, Ty const&>{data[left], data[right]};
    }

    [[nodiscard]] inline total_t const& raw() const noexcept {
      return matches_;
    }

    [[nodiscard]] inline std::optional<cdt::offset_t> offset() const noexcept {
      if (matches_.empty()) {
        return {};
      }

      auto best{std::max_element(
          matches_.begin(), matches_.end(), [](auto& lhs, auto& rhs) {
            return match_value{lhs.second} < match_value{rhs.second};
          })};

      return best->first;
    }

  private:
    couple_t couple_;
    total_t matches_;
  };

  template<std::uint8_t Depth>
  [[nodiscard]] auto build_deltas(std::vector<snippet<Depth>> const& snippets) {
    std::vector<details::delta> deltas;

    auto segments{snippets.size()};
    deltas.reserve(segments * (segments - 1) / 2);

    for (std::int16_t i{0}; i < segments; ++i) {
      for (std::int16_t j{i + 1}; j < segments; ++j) {
        deltas.emplace_back(couple_t{i, j});
      }
    }

    return deltas;
  }

  template<std::uint8_t Depth>
  void match_all(std::vector<snippet<Depth>> const& snippets,
                 std::vector<delta>& deltas) {
    std::for_each(std::execution::par,
                  deltas.begin(),
                  deltas.end(),
                  [&snippets](auto& d) {
                    auto [left, right]{d.couple(snippets)};

                    auto ticket{kpm::match(details::match_config{},
                                           left.grid_[0],
                                           left.mask_,
                                           right.grid_[0],
                                           right.mask_)};
                    d.update_raw(ticket);
                  });
  }

  void crossmatch_single(delta& pi, delta& pj, delta& pk) noexcept {
    for (auto& [i, u1] : pi.raw()) {
      auto& [ix, iy]{i};
      for (auto& [j, u2] : pj.raw()) {
        auto& [jx, jy]{j};
        for (auto& [k, u3] : pk.raw()) {
          auto& [kx, ky]{k};
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
      auto [ix, iy]{i->couple()};

      for (auto j{i + 1}; j < deltas.end(); ++j) {
        if (auto [jx, jy]{j->couple()}; jx == ix) {
          auto k{deltas[iy * (2 * segments - iy - 1) / 2 + jy - iy - 1]};

          crossmatch_single(*i, *j, k);
        }
        else {
          break;
        }
      }
    }
  }

  template<typename Ty>
  concept walker =
      std::invocable<Ty> && std::invocable<Ty, std::uint16_t, cdt::offset_t>;

  class graph {
  private:
    struct node;

    struct edge {
      std::uint16_t link_;
      cdt::offset_t offset_;
    };

    struct node {
      bool visited_;
      std::vector<edge> edges_;
    };

    struct state {
      std::uint16_t node_{};
      std::uint16_t edge_{};
      cdt::offset_t offset_{};
    };

    using history_t = std::stack<state>;

  public:
    graph(std::uint16_t size)
        : nodes_{size, node{}} {
    }

    void add_edge(couple_t couple, cdt::offset_t const& offset) {
      auto [idx, jdx]{couple};

      nodes_[idx].edges_.emplace_back(jdx, offset);
      nodes_[jdx].edges_.emplace_back(idx, -offset);
    }

    template<walker Walker>
    void process(Walker& w) {
      for (std::uint16_t i{0}; i < nodes_.size(); ++i) {
        if (!nodes_[i].visited_) {
          w();

          walk({i, 0, {}}, w);
        }
      }
    }

  private:
    template<walker Walker>
    void walk(state active, Walker& w) {
      for (history_t hist; true;) {
        if (auto& current{nodes_[active.node_]}; current.visited_) {
          if (!backtrack(active, hist)) {
            break;
          }
        }
        else {
          current.visited_ = true;

          w(active.node_, active.offset_);

          if (active.edge_ < current.edges_.size()) {
            hist.emplace(active.node_, active.edge_ + 1, active.offset_);
            active = *advance(active, current);
          }
          else if (!backtrack(active, hist)) {
            break;
          }
        }
      }
    }

    [[nodiscard]] inline bool backtrack(state& active,
                                        history_t& hist) noexcept {
      while (!hist.empty()) {
        auto next{advance(hist.top())};
        hist.pop();

        if (next) {
          active = *next;
          return true;
        }
      }

      return false;
    }

    [[nodiscard]] inline std::optional<state>
        advance(state const& s) const noexcept {
      return advance(s, nodes_[s.node_]);
    }

    [[nodiscard]] inline std::optional<state>
        advance(state const& s, node const& current) const noexcept {
      if (s.edge_ < current.edges_.size()) {
        auto& next{current.edges_[s.edge_]};
        return std::optional<state>{
            std::in_place, next.link_, s.edge_, s.offset_ + next.offset_};
      }

      return {};
    }

  private:
    std::vector<node> nodes_;
  };

  template<std::uint8_t Depth>
  class splicer {
  public:
    static inline constexpr auto depth{Depth};

    using fragment_t = fgm::fragment<depth>;

    using snippet_type = snippet<depth>;

  public:
    inline explicit splicer(std::vector<snippet_type>&& snippets) noexcept
        : snippets_{std::move(snippets)} {
    }

    inline void operator()() {
      result_.emplace_back();
    }

    inline void operator()(std::uint16_t snippet, cdt::offset_t offset) {
      result_.back().blit(offset, std::move(snippets_[snippet].fragment_));
    }

    [[nodiscard]] inline std::list<fragment_t> get_result() noexcept {
      return std::move(result_);
    }

  private:
    std::vector<snippet_type> snippets_;
    std::list<fragment_t> result_;
  };

  [[nodiscard]] graph build_graph(std::vector<delta> const& deltas,
                                  std::uint16_t size) {
    graph result{size};
    for (auto& d : deltas) {
      if (auto off{d.offset()}; off) {
        result.add_edge(d.couple(), *off);
      }
    }

    return result;
  }
} // namespace details

template<std::uint8_t Depth, typename Iter>
[[nodiscard]] std::list<fgm::fragment<Depth>> splice(Iter first, Iter last) {
  auto snippets{details::extract_all<Depth>(first, last)};
  auto deltas{details::build_deltas(snippets)};
  auto size{snippets.size()};

  details::match_all(snippets, deltas);
  details::crossmatch_all(deltas, size);

  details::splicer<Depth> spliced{std::move(snippets)};
  build_graph(deltas, size).process(spliced);

  return spliced.get_result();
}
} // namespace fgs
