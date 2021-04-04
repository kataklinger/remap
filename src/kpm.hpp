
#pragma once

#include "all.hpp"
#include "cdt.hpp"
#include "kpr.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kpm {
template<typename Ty>
concept match_config = requires(Ty cfg) {
  requires std::unsigned_integral<decltype(Ty::weight_switch)>;

  requires std::unsigned_integral<decltype(Ty::region_votes)>;
  requires Ty::region_votes >= 1;

  typename Ty::allocator_type;

  { cfg.get_allocator() }
  ->std::convertible_to<typename Ty::allocator_type>;
};

template<match_config Cfg, typename Ty>
using get_allocator = all::rebind_alloc_t<typename Cfg::allocator_type, Ty>;

struct vote {
  using pair_t = std::pair<cdt::offset_t const, std::size_t>;

  inline vote() noexcept = default;

  inline vote(pair_t const& pair) noexcept
      : offset_{std::get<0>(pair)}
      , count_{std::get<1>(pair)} {
  }

  inline vote& operator=(pair_t const& pair) noexcept {
    offset_ = std::get<0>(pair);
    count_ = std::get<1>(pair);

    return *this;
  }

  cdt::offset_t offset_{};
  std::size_t count_{};
};

template<match_config Cfg>
using ticket_t = std::vector<vote, get_allocator<Cfg, vote>>;

template<match_config Cfg>
using totalizator_t = std::unordered_map<cdt::offset_t,
                                         std::size_t,
                                         cdt::offset_hash,
                                         std::equal_to<cdt::offset_t>,
                                         get_allocator<Cfg, vote::pair_t>>;

namespace details {
  template<match_config Cfg>
  using collector_t =
      std::vector<ticket_t<Cfg>, get_allocator<Cfg, ticket_t<Cfg>>>;

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
  [[nodiscard]] totalizator_t<Cfg> count_offsets(Cfg const& config,
                                                 Region const& previous,
                                                 Region const& current) {
    totalizator_t<Cfg> total{config.get_allocator()};

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
  [[nodiscard]] ticket_t<Cfg> top_offsets(Cfg const& config,
                                          totalizator_t<Cfg> const& total,
                                          std::size_t top) {
    ticket_t<Cfg> selected{top + 1, config.get_allocator()};

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
  [[nodiscard]] inline ticket_t<Cfg>
      cast_vote_impl(Cfg const& config,
                     Region const& previous,
                     Region const& current,
                     std::bool_constant<Switch> /*unused*/) {
    return top_offsets(config,
                       count_offsets<Switch>(config, previous, current),
                       Cfg::region_votes);
  }

  template<match_config Cfg>
  [[nodiscard]] totalizator_t<Cfg> count(collector_t<Cfg> const& tickets) {
    totalizator_t<Cfg> total{tickets.get_allocator()};

    for (auto& ticket : tickets) {
      auto rank{Cfg::region_votes};
      for (auto& [off, cnt] : ticket) {
        total[off] += rank--;
      }
    }

    return total;
  }

  template<match_config Cfg>
  [[nodiscard]] std::optional<cdt::offset_t> declare(ticket_t<Cfg> const& top,
                                                     std::size_t region_count) {
    if (top.empty()) {
      return {};
    }

    if (top.size() > 1 && top[0].count_ < top[1].count_ + region_count / 2) {
      return {};
    }

    return {top[0].offset_};
  }

  template<match_config Cfg, typename Region>
  [[nodiscard]] inline ticket_t<Cfg> cast_vote(Cfg const& config,
                                               Region const& previous,
                                               Region const& current) {
    constexpr auto idx{Region::max_weight - 1};

    return previous.counts()[idx] < Cfg::weight_switch ||
                   current.counts()[idx] <= Cfg::weight_switch
               ? cast_vote_impl(config, previous, current, std::true_type{})
               : cast_vote_impl(config, previous, current, std::false_type{});
  }

  template<typename Region>
  inline std::size_t intersect_count(Region const& region,
                                     mrl::region_t limits) {
    std::size_t count{};

    for (auto const& [c, group] : region.points()) {
      for (auto point : group) {
        if (limits.contains(point)) {
          ++count;
        }
      }
    }

    return count;
  }

  using intersect_t = std::pair<mrl::limits_t, mrl::limits_t>;

  [[nodiscard]] inline intersect_t get_limits(std::int32_t delta,
                                              mrl::size_type previous,
                                              mrl::size_type current) noexcept {
    return delta < 0
               ? intersect_t{{0, previous - delta}, {0ULL + delta, current}}
               : intersect_t{{0ULL + delta, previous}, {0, current - delta}};
  }

  template<typename Region>
  [[nodiscard]] inline bool is_overlapped(mrl::limits_t const& hor,
                                          mrl::limits_t const& ver,
                                          mrl::dimensions_t const& dim,
                                          Region const& region,
                                          std::size_t matches) {
    constexpr auto min_density{1.0f / (32 * 32)};

    auto overlap_region{from_limits(hor, ver)};
    auto overlap_count{details::intersect_count(region, overlap_region)};

    auto overlap_area{static_cast<float>(overlap_region.area())};
    auto area_rate{overlap_area / dim.area()};

    auto overlap_rate{static_cast<float>(overlap_count) / region.total_count()};

    auto match_rate{static_cast<float>(matches) / overlap_count};
    auto match_density{matches / overlap_area};

    return area_rate >= 0.015f && match_density >= min_density &&
           match_rate >= 1 - 0.3f * std::hypotf(overlap_rate, 1 - area_rate);
  }

} // namespace details

template<match_config Cfg, typename Region>
[[nodiscard]] inline ticket_t<Cfg> match(Cfg const& config,
                                         Region const& previous,
                                         mrl::dimensions_t pdim,
                                         Region const& current,
                                         mrl::dimensions_t cdim) {
  using namespace details;

  auto ticket{cast_vote(config, previous, current)};

  ticket.erase(std::remove_if(ticket.begin(), ticket.end(), [&](auto const& v) {
    auto hor{get_limits(v.offset_.x_, pdim.width_, cdim.width_)};
    auto ver{get_limits(v.offset_.y_, pdim.height_, cdim.height_)};

    return !is_overlapped(hor.first, ver.first, pdim, previous, v.count_) ||
           !is_overlapped(hor.second, ver.second, cdim, current, v.count_);
  }));

  return ticket;
}

template<match_config Cfg,
         typename Alloc,
         std::size_t Width,
         std::size_t Height>
[[nodiscard]] std::optional<cdt::offset_t>
    match(Cfg const& config,
          kpr::grid<Width, Height, Alloc> const& previous,
          kpr::grid<Width, Height, Alloc> const& current) {
  using namespace details;

  using gird_t = kpr::grid<Width, Height, Alloc>;

  collector_t<Cfg> tickets{config.get_allocator()};
  tickets.reserve(gird_t::region_count);

  auto prev_regs{previous.regions()}, curr_regs{current.regions()};

  for (std::size_t i{0}; i < gird_t::region_count; ++i) {
    tickets.push_back(cast_vote(config, prev_regs[i], curr_regs[i]));
  }

  return declare<Cfg>(top_offsets(config, count<Cfg>(tickets), 2),
                      gird_t::region_count);
}
} // namespace kpm
