
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

using vote_t = std::tuple<cdt::offset_t, std::size_t>;

template<match_config Cfg>
using ticket_t = std::vector<vote_t, get_allocator<Cfg, vote_t>>;

template<match_config Cfg>
using totalizator_t = std::unordered_map<
    cdt::offset_t,
    std::size_t,
    cdt::offset_hash,
    std::equal_to<cdt::offset_t>,
    get_allocator<Cfg, std::pair<const cdt::offset_t, std::size_t>>>;

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
        if (std::get<1>(added) > std::get<1>(current)) {
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
      vote_impl(Cfg const& config,
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

    if (top.size() > 1 &&
        std::get<1>(top[0]) < std::get<1>(top[1]) + region_count / 2) {
      return {};
    }

    return {std::get<0>(top[0])};
  }

} // namespace details

template<match_config Cfg, typename Region>
[[nodiscard]] inline ticket_t<Cfg>
    vote(Cfg const& config, Region const& previous, Region const& current) {
  return previous.counts()[2] < Cfg::weight_switch ||
                 current.counts()[2] <= Cfg::weight_switch
             ? details::vote_impl(config, previous, current, std::true_type{})
             : details::vote_impl(config, previous, current, std::false_type{});
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
    tickets.push_back(vote(config, prev_regs[i], curr_regs[i]));
  }

  return declare<Cfg>(top_offsets(config, count<Cfg>(tickets), 2),
                      gird_t::region_count);
}
} // namespace kpm
