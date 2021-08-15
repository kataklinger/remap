
// keypoint matching

#pragma once

#include "all.hpp"
#include "cdt.hpp"
#include "kpr.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kpm {

template<typename Ty>
concept match_config = requires(Ty cfg) {
  requires std::unsigned_integral<decltype(Ty::weight_switch)>;

  requires std::unsigned_integral<decltype(Ty::region_votes)>;
  requires Ty::region_votes >= 1;

  typename Ty::allocator_type;

  { cfg.get_allocator() } -> std::convertible_to<typename Ty::allocator_type>;
};

template<match_config Cfg>
using config_alloc_t = typename Cfg::allocator_type;

struct vote {
  using pair_t = std::pair<cdt::offset_t const, std::size_t>;

  inline vote() noexcept = default;

  inline vote(cdt::offset_t const& offset, std::size_t count) noexcept
      : offset_{offset}
      , count_{count} {
  }

  inline explicit vote(pair_t const& pair) noexcept
      : offset_{std::get<0>(pair)}
      , count_{std::get<1>(pair)} {
  }

  inline vote& operator=(pair_t const& pair) noexcept {
    offset_ = std::get<0>(pair);
    count_ = std::get<1>(pair);

    return *this;
  }

  [[nodiscard]] inline vote reverse() const noexcept {
    return {-offset_, count_};
  }

  cdt::offset_t offset_{};
  std::size_t count_{};
};

template<typename Alloc>
using ticket_t = std::vector<vote, all::rebind_alloc_t<Alloc, vote>>;

template<typename Alloc>
using totalizator_t =
    std::unordered_map<cdt::offset_t,
                       std::size_t,
                       cdt::offset_hash,
                       std::equal_to<cdt::offset_t>,
                       all::rebind_alloc_t<Alloc, vote::pair_t>>;

using cellular_totalizator_t =
    std::unordered_map<cdt::offset_t,
                       totalizator_t<std::allocator<char>>,
                       cdt::offset_hash>;

using cell_size_t = cdt::dimensions<std::uint8_t>;

namespace details {

  template<typename Alloc>
  using collector_t =
      std::vector<ticket_t<Alloc>, all::rebind_alloc_t<Alloc, ticket_t<Alloc>>>;

  template<typename Total, typename Points>
  void
      get_offsets(Points const& previous, Points const& current, Total& total) {
    for (auto& [px, py] : previous) {
      for (auto& [cx, cy] : current) {
        cdt::offset_t off{
            static_cast<std::int32_t>(px) - static_cast<std::int32_t>(cx),
            static_cast<std::int32_t>(py) - static_cast<std::int32_t>(cy)};

        ++total[off];
      }
    }
  }

  template<bool Switch, match_config Cfg, typename Region>
  [[nodiscard]] auto count_offsets(Cfg const& config,
                                   Region const& previous,
                                   Region const& current) {
    totalizator_t<config_alloc_t<Cfg>> total{config.get_allocator()};

    auto& prev_group{previous.points()};
    for (auto& [key, curr] : current.points()) {
      if constexpr (!Switch) {
        if (kpr::weight(key) != std::byte{2}) {
          continue;
        }
      }

      if (auto it = prev_group.find(key); it != prev_group.end()) {
        get_offsets(it->second, curr, total);
      }
    }

    return total;
  }

  template<match_config Cfg>
  [[nodiscard]] auto
      top_offsets(Cfg const& config,
                  totalizator_t<config_alloc_t<Cfg>> const& total,
                  std::size_t top) {
    ticket_t<config_alloc_t<Cfg>> selected{top + 1, config.get_allocator()};

    for (auto& added : total) {
      auto i{top};
      while (i != 0) {
        auto& current = selected[--i];
        if (added.second > current.count_) {
          selected[i + 1] = current;
          if (i == 0) {
            selected[i] = added;
            break;
          }
        }
        else {
          if (i < top - 1) {
            selected[i + 1] = added;
          }
          break;
        }
      }
    }

    if (auto limit{std::min(top, total.size())}; selected.size() > limit) {
      selected.resize(limit);
    }

    return selected;
  }

  template<match_config Cfg, typename Region, bool Switch>
  [[nodiscard]] inline auto
      cast_vote_impl(Cfg const& config,
                     Region const& previous,
                     Region const& current,
                     std::bool_constant<Switch> /*unused*/) {
    return top_offsets(config,
                       count_offsets<Switch>(config, previous, current),
                       Cfg::region_votes);
  }

  template<match_config Cfg>
  [[nodiscard]] auto count(collector_t<config_alloc_t<Cfg>> const& tickets) {
    totalizator_t<config_alloc_t<Cfg>> total{tickets.get_allocator()};

    for (auto& ticket : tickets) {
      auto rank{Cfg::region_votes};
      for (auto& [off, cnt] : ticket) {
        total[off] += rank--;
      }
    }

    return total;
  }

  template<typename Alloc, std::size_t Width, std::size_t Height>
  [[nodiscard]] inline std::size_t
      get_active(kpr::grid<Width, Height, Alloc> const& grid) noexcept {
    std::size_t count{0};
    for (auto& region : grid.regions()) {
      if (region.is_active()) {
        ++count;
      }
    }

    return count;
  }

  template<typename Tickets>
  [[nodiscard]] inline std::optional<cdt::offset_t>
      declare(Tickets const& top, std::size_t region_count) {
    if (top.empty()) {
      return {};
    }

    if (top.size() > 1 && top[0].count_ < top[1].count_ + region_count / 2) {
      return {};
    }

    return {top[0].offset_};
  }

  template<match_config Cfg, typename Region>
  [[nodiscard]] inline auto cast_vote(Cfg const& config,
                                      Region const& previous,
                                      Region const& current) {
    constexpr auto idx{Region::max_weight - 1};

    return previous.counts()[idx] < Cfg::weight_switch ||
                   current.counts()[idx] <= Cfg::weight_switch
               ? cast_vote_impl(config, previous, current, std::true_type{})
               : cast_vote_impl(config, previous, current, std::false_type{});
  }

  [[nodiscard]] inline std::int32_t to_cell(std::int32_t pval,
                                            std::int32_t cval,
                                            std::uint8_t size) noexcept {
    return std::min(pval, cval) / size;
  }

  template<typename Total, typename Points>
  void get_offsets(Points const& previous,
                   Points const& current,
                   Total& total,
                   cell_size_t const& cell_size) {
    auto& [dx, dy]{cell_size};

    for (auto& [px, py] : previous) {
      for (auto& [cx, cy] : current) {
        auto ox{static_cast<std::int32_t>(px) - static_cast<std::int32_t>(cx)};
        auto oy{static_cast<std::int32_t>(py) - static_cast<std::int32_t>(cy)};

        auto [it, added]{total.try_emplace({ox, oy}, total.get_allocator())};
        ++(it->second)[{to_cell(px, cx, dx), to_cell(py, cy, dy)}];
      }
    }
  }

  template<typename Region>
  [[nodiscard]] cellular_totalizator_t
      count_offsets(Region const& previous,
                    Region const& current,
                    cell_size_t const& cell_size) {
    cellular_totalizator_t total;

    auto& prev_group{previous.points()};
    for (auto& [key, curr] : current.points()) {
      if (auto it = prev_group.find(key); it != prev_group.end()) {
        get_offsets(it->second, curr, total, cell_size);
      }
    }

    return total;
  }

  struct best_offset {
    cdt::offset_t offset_;
    std::size_t matched_cells_;
    std::size_t matched_keypoints_;

    [[nodiscard]] inline vote as_vote() const noexcept {
      return {offset_, matched_keypoints_};
    }

    inline friend auto operator<=>(best_offset const& lhs,
                                   best_offset const& rhs) noexcept {
      return lhs.matched_keypoints_ <=> rhs.matched_keypoints_;
    }
  };

  [[nodiscard]] best_offset find_best(cellular_totalizator_t const& offsets) {
    std::vector<best_offset> scores;
    scores.reserve(offsets.size());

    transform(
        begin(offsets), end(offsets), back_inserter(scores), [](auto& item) {
          return best_offset{item.first,
                             item.second.size(),
                             accumulate(begin(item.second),
                                        end(item.second),
                                        std::size_t{},
                                        [](auto const& total, auto& value) {
                                          return total + get<1>(value);
                                        })};
        });

    return *max_element(begin(scores), end(scores));
  }

  using intersect_span = std::pair<mrl::limits_t, mrl::limits_t>;

  [[nodiscard]] inline intersect_span
      get_limits(std::int32_t delta,
                 mrl::size_type previous,
                 mrl::size_type current) noexcept {
    if (delta < 0) {
      delta = std::abs(delta);
      return intersect_span{
          {0, std::min(previous, current - delta)},
          {0ULL + delta, std::min(current, previous + delta)}};
    }

    return intersect_span{{0ULL + delta, std::min(previous, current + delta)},
                          {0, std::min(current, previous - delta)}};
  }

  template<typename Region>
  [[nodiscard]] std::size_t
      filter_keypoints(Region const& region,
                       sid::mon::dimg_t const& mask,
                       cdt::offset_t delta,
                       mrl::region_t limits,
                       cell_size_t const& cell_size) noexcept {
    std::unordered_set<
        cdt::offset_t,
        cdt::offset_hash,
        std::equal_to<cdt::offset_t>,
        all::rebind_alloc_t<typename Region::allocator_type, cdt::offset_t>>
        cells{region.get_allocator()};

    auto& [cx, cy]{cell_size};
    for (auto const& [key, group] : region.points()) {
      for (auto point : group) {
        if (limits.contains(point)) {
          if (auto idx{to_index(static_cast<cdt::offset_t>(point) + delta,
                                mask.dimensions())};
              value(mask.data()[idx]) != 0) {
            auto ox{static_cast<std::int32_t>(point.x_ - limits.left_) / cx};
            auto oy{static_cast<std::int32_t>(point.y_ - limits.top_) / cy};

            cells.emplace(ox * cx, oy * cy);
          }
        }
      }
    }

    return cells.size();
  }

  template<typename Region>
  [[nodiscard]] std::size_t count_active_cells(Region const& preg,
                                               sid::mon::dimg_t const& pmask,
                                               Region const& creg,
                                               sid::mon::dimg_t const& cmask,
                                               cdt::offset_t const& offset,
                                               cell_size_t const& cell_size) {
    auto& [x, y]{offset};

    auto &pdim{pmask.dimensions()}, &cdim{cmask.dimensions()};
    auto hor{get_limits(x, pdim.width_, cdim.width_)},
        ver{get_limits(y, pdim.height_, cdim.height_)};

    auto plim{from_limits(hor.first, ver.first)},
        clim{from_limits(hor.second, ver.second)};

    return filter_keypoints(creg, pmask, offset, clim, cell_size);
  }

} // namespace details

template<typename Region>
[[nodiscard]] std::optional<vote> match(Region const& preg,
                                        sid::mon::dimg_t const& pmask,
                                        Region const& creg,
                                        sid::mon::dimg_t const& cmask,
                                        cell_size_t const& cell_size) {
  using namespace details;

  auto offsets{count_offsets(preg, creg, cell_size)};
  if (offsets.empty()) {
    return {};
  }

  auto best{find_best(offsets)};

  auto active{
      count_active_cells(preg, pmask, creg, cmask, best.offset_, cell_size)};
  if (best.matched_cells_ < active * 0.66f) {
    return {};
  }

  return best.as_vote();
}

template<match_config Cfg, typename Grid>
[[nodiscard]] std::optional<cdt::offset_t>
    match(Cfg const& config, Grid const& previous, Grid const& current) {
  using namespace details;

  auto active{get_active(current)};
  if (active < Grid::region_count / 4) {
    return {};
  }

  collector_t<config_alloc_t<Cfg>> tickets{config.get_allocator()};
  tickets.reserve(Grid::region_count);

  auto prev_regs{previous.regions()}, curr_regs{current.regions()};

  for (std::size_t i{0}; i < Grid::region_count; ++i) {
    tickets.push_back(cast_vote(config, prev_regs[i], curr_regs[i]));
  }

  return declare(top_offsets(config, count<Cfg>(tickets), 2), active);
}

} // namespace kpm
