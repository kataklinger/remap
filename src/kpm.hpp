
#pragma once

#include "kpr.hpp"

#include <array>
#include <concepts>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kpm {
using offset = std::tuple<std::int32_t, std::int32_t>;

struct offset_hash {
  [[nodiscard]] std::size_t operator()(offset const& off) const noexcept {
    std::size_t hashed = 2166136261U;

    hashed ^= static_cast<std::size_t>(std::get<0>(off));
    hashed *= 16777619U;

    hashed ^= static_cast<std::size_t>(std::get<1>(off));
    hashed *= 16777619U;

    return hashed;
  }
};

template<typename Ty>
concept match_config = requires(Ty cfg) {
  requires std::unsigned_integral<decltype(Ty::weight_switch)>;

  requires std::unsigned_integral<decltype(Ty::region_votes)>;
  requires Ty::region_votes >= 1;

  typename Ty::allocator_type;
};

namespace details {
  template<match_config Cfg, typename Ty>
  using get_allocator = typename std::allocator_traits<
      typename Cfg::allocator_type>::template rebind_alloc<Ty>;

  template<match_config Cfg>
  using totalizator_t = std::unordered_map<
      offset,
      std::size_t,
      offset_hash,
      std::equal_to<offset>,
      get_allocator<Cfg, std::pair<const offset, std::size_t>>>;

  template<match_config Cfg>
  using ticket_t = std::vector<offset, get_allocator<Cfg, offset>>;

  template<match_config Cfg>
  using collector_t =
      std::vector<ticket_t<Cfg>, get_allocator<Cfg, ticket_t<Cfg>>>;

  template<typename Total>
  void get_offsets(kpr::region::points_t const& previous,
                   kpr::region::points_t const& current,
                   Total& total) {
    for (auto& [px, py] : previous) {
      for (auto& [cx, cy] : current) {
        offset off{
            static_cast<std::int32_t>(px) - static_cast<std::int32_t>(cx),
            static_cast<std::int32_t>(py) - static_cast<std::int32_t>(cy)};

        ++total[off];
      }
    }
  }

  template<bool Switch, match_config Cfg>
  totalizator_t<Cfg> count_offsets(Cfg& config,
                                   kpr::region const& previous,
                                   kpr::region const& current) {
    totalizator_t<Cfg> total;

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
  ticket_t<Cfg> top_offsets(Cfg& config, totalizator_t<Cfg> const& total) {
    ticket_t<Cfg> selected{Cfg::region_votes + 1};
    std::vector<std::size_t> counts{Cfg::region_votes + 1};

    for (auto& [off, cnt] : total) {
      std::size_t i{Cfg::region_votes};
      while (i != 0) {
        if (cnt > counts[--i]) {
          counts[i + 1] = counts[i];
          selected[i + 1] = selected[i];
        }
        else {
          break;
        }
      }

      if (i < Cfg::region_votes - 1) {
        counts[i] = cnt;
        selected[i] = off;
      }
    }

    if (selected.size() > Cfg::region_votes) {
      selected.resize(Cfg::region_votes);
    }

    return selected;
  }

  template<match_config Cfg, bool Switch>
  inline ticket_t<Cfg> vote_helper(Cfg& config,
                                   kpr::region const& previous,
                                   kpr::region const& current,
                                   std::bool_constant<Switch> /*unused*/) {
    return top_offsets(config,
                       count_offsets<Switch>(config, previous, current));
  }

  template<match_config Cfg>
  inline ticket_t<Cfg> vote(Cfg& config,
                            kpr::region const& previous,
                            kpr::region const& current) {
    return previous.counts()[2] < Cfg::weight_switch ||
                   current.counts()[2] <= Cfg::weight_switch
               ? vote_helper(config, previous, current, std::true_type{})
               : vote_helper(config, previous, current, std::false_type{});
  }

  template<match_config Cfg>
  std::optional<offset> count(Cfg& config, collector_t<Cfg> const& tickets) {
    totalizator_t<Cfg> total;

    for (auto& ticket : tickets) {
      auto rank{Cfg::region_votes};
      for (auto& vote : ticket) {
        total[vote] += rank--;
      }
    }

    if (!total.empty()) {
      auto& [off, cnt] = *std::max_element(
          total.begin(), total.end(), [](auto& lhs, auto& rhs) {
            return lhs.second < rhs.second;
          });
    }

    return {};
  }
} // namespace details

template<match_config Cfg, std::size_t Width, std::size_t Height>
std::optional<offset> match(Cfg& config,
                            kpr::grid<Width, Height> const& previous,
                            kpr::grid<Width, Height> const& current) {
  using gird_t = kpr::grid<Width, Height>;

  details::collector_t<Cfg> tickets;
  tickets.reserve(gird_t::region_count);

  auto &prev_regs{previous.regions()}, &curr_regs{current.regions()};

  for (std::size_t i{0}; i < gird_t::region_count; ++i) {
    tickets.push_back(details::vote(config, prev_regs[i], curr_regs[i]));
  }

  return details::count(config, tickets);
}
} // namespace kpm
