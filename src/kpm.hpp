
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

  struct intersect_data {
    std::size_t count_;
    float rate_;
  };

  template<typename Region>
  [[nodiscard]] intersect_data
      filter_keypoints(Region const& region,
                       mrl::matrix<std::uint8_t> const& mask,
                       cdt::offset_t delta,
                       mrl::region_t limits) noexcept {
    std::size_t count{};

    for (auto const& [c, group] : region.points()) {
      for (auto point : group) {
        if (limits.contains(point)) {
          if (auto idx{to_index(static_cast<cdt::offset_t>(point) + delta,
                                mask.dimensions())};
              mask.data()[idx] != 0) {
            ++count;
          }
        }
      }
    }

    return {count, static_cast<float>(count) / region.total_count()};
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

  struct overlap_data {
    inline overlap_data(float match_area,
                        float total_area,
                        float match_count,
                        float total_count,
                        intersect_data intersect) noexcept
        : overlap_keypoints_count_{intersect.count_}
        , overlap_keypoints_rate_{intersect.count_ / total_count}
        , overlap_area_rate_{match_area / total_area}
        , match_keypoints_rate_{match_count / intersect.count_}
        , match_keypoints_density_{match_count / match_area} {
    }

    std::size_t overlap_keypoints_count_;

    float overlap_keypoints_rate_;
    float match_keypoints_rate_;

    float match_keypoints_density_;

    float overlap_area_rate_;
  };

  [[nodiscard]] bool is_overlapping(overlap_data const& fst,
                                    overlap_data const& snd) noexcept {
    constexpr auto min_density{1.0f / (32 * 32)};

    auto overlap_rate{
        std::max(fst.overlap_keypoints_rate_, snd.overlap_keypoints_rate_)};

    auto area_rate{std::max(fst.overlap_area_rate_, snd.overlap_area_rate_)};

    auto match_rate{
        std::max(fst.match_keypoints_rate_, snd.match_keypoints_rate_)};

    return area_rate >= 0.015f && fst.match_keypoints_density_ >= min_density &&
           match_rate >= 1 - 0.35f * std::hypotf(overlap_rate, 1 - area_rate);
  }
} // namespace details

template<match_config Cfg, typename Region>
[[nodiscard]] inline ticket_t<Cfg>
    match(Cfg const& config,
          Region const& preg,
          mrl::matrix<std::uint8_t> const& pmask,
          Region const& creg,
          mrl::matrix<std::uint8_t> const& cmask) {
  using namespace details;

  auto ticket{cast_vote(config, preg, creg)};

  auto it{std::remove_if(ticket.begin(), ticket.end(), [&](auto const& v) {
    auto &pdim{pmask.dimensions()}, &cdim{cmask.dimensions()};

    auto hor{get_limits(v.offset_.x_, pdim.width_, cdim.width_)},
        ver{get_limits(v.offset_.y_, pdim.height_, cdim.height_)};

    auto plim{from_limits(hor.first, ver.first)},
        clim{from_limits(hor.second, ver.second)};

    details::overlap_data pover(
        plim.area(),
        pmask.dimensions().area(),
        v.count_,
        preg.total_count(),
        filter_keypoints(preg, cmask, -v.offset_, plim));

    details::overlap_data cover(clim.area(),
                                cmask.dimensions().area(),
                                v.count_,
                                creg.total_count(),
                                filter_keypoints(creg, pmask, v.offset_, clim));

    return !details::is_overlapping(pover, cover);
  })};

  ticket.erase(it, ticket.end());

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
