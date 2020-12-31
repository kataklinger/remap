
#pragma once

#include "mrl.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace kpr {
static inline constexpr uint8_t code_length = 13;
static inline constexpr uint8_t code_max_index = code_length - 1;

using code = std::array<std::byte, code_length>;
using point = std::tuple<mrl::size_type, mrl::size_type>;

[[nodiscard]] inline std::byte weight(code const& value) noexcept {
  return value[code_max_index] & static_cast<std::byte>(0xf);
}

struct code_hash {
  [[nodiscard]] std::size_t operator()(code const& key) const noexcept {
    std::size_t hashed = 2166136261U;
    for (auto value : key) {
      hashed ^= static_cast<std::size_t>(value);
      hashed *= 16777619U;
    }
    return hashed;
  }
};

namespace details {
  template<typename Idxs, typename Jdxs>
  struct join_sequences_help;

  template<std::size_t... Idxs, std::size_t... Jdxs>
  struct join_sequences_help<std::index_sequence<Idxs...>,
                             std::index_sequence<Jdxs...>> {
    using type = std::index_sequence<Idxs..., Jdxs...>;
  };

  template<typename... Seqs>
  struct join_sequences;

  template<std::size_t... Idxs>
  struct join_sequences<std::index_sequence<Idxs...>> {
    using type = std::index_sequence<Idxs...>;
  };

  template<typename Seq, typename... Rest>
  struct join_sequences<Seq, Rest...> {
    using type = typename join_sequences_help<
        Seq,
        typename join_sequences<Rest...>::type>::type;
  };

  template<typename... Seqs>
  using join_sequences_t = typename join_sequences<Seqs...>::type;

  template<std::size_t InSize, std::size_t Outer, typename Inner>
  struct offset_inner;

  template<std::size_t InSize, std::size_t Outer, std::size_t... Inners>
  struct offset_inner<InSize, Outer, std::index_sequence<Inners...>> {
    using type = std::index_sequence<(InSize * Outer + Inners)...>;
  };

  template<std::size_t InSize, std::size_t Outer, typename Inner>
  using offset_inner_t = typename offset_inner<InSize, Outer, Inner>::type;

  template<std::size_t InSize, typename Outer, typename Inner>
  struct explode;

  template<std::size_t InSize, std::size_t... Outers, typename Inner>
  struct explode<InSize, std::index_sequence<Outers...>, Inner> {
    using type = join_sequences_t<offset_inner_t<InSize, Outers, Inner>...>;
  };

} // namespace details

template<std::size_t InSize, typename Outer, typename Inner>
using grid_explode = typename details::
    explode<InSize, std::decay_t<Outer>, std::decay_t<Inner>>::type;

class region {
public:
  static inline constexpr std::size_t max_weight{3};

  using points_store =
      std::unordered_multimap<code, point, code_hash, std::equal_to<code>>;
  using count_store = std::array<std::size_t, max_weight>;

public:
  inline void add(std::pair<code, point>&& point) {
    auto w{weight(point.first)};

    points_.insert(std::move(point));

    ++weight_count_[static_cast<std::size_t>(w)];
  }

  inline void clear() noexcept {
    points_.clear();
  }

  [[nodiscard]] inline points_store const& points() const noexcept {
    return points_;
  }

  [[nodiscard]] inline count_store const& counts() const noexcept {
    return weight_count_;
  }

private:
  points_store points_;
  count_store weight_count_;
};

template<std::size_t Width, std::size_t Height>
class grid {
  static_assert(Width > 0);
  static_assert(Height > 0);

public:
  static inline constexpr std::size_t width{Width};
  static inline constexpr std::size_t height{Height};
  static inline constexpr std::size_t region_count{width * height};

  using region_store = std::array<region, region_count>;

public:
  template<std::size_t... Idxs>
  inline void
      add(code const& key, point pt, std::index_sequence<Idxs...> /**/) {
    add_intern(key, pt, std::integral_constant<std::size_t, Idxs>{}...);
  }

  inline void clear() noexcept {
    for (auto& region : regions_) {
      region.clear();
    }
  }

  [[nodiscard]] inline region_store const& regions() const noexcept {
    return regions_;
  }

private:
  template<typename... Idxs>
  inline void add_intern(code const& key, point pt, Idxs... idxs) {
    ((( void )add_intern(key, pt, idxs)), ...);
  }

  template<std::size_t Idx>
  inline void add_intern(code const& key,
                         point pt,
                         std::integral_constant<std::size_t, Idx> /*unused*/) {
    regions_[Idx].add({key, pt});
  }

private:
  region_store regions_;
};

template<typename Ty>
concept grid_size = std::unsigned_integral<std::decay_t<Ty>>;

template<typename Ty>
concept gridlike = requires(Ty g) {
  { Ty::width }
  ->grid_size<>;
  { Ty::height }
  ->grid_size<>;
  {g.clear()};
};

} // namespace kpr
