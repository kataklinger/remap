
#pragma once

#include "mrl.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
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

template<typename Alloc>
class region {
public:
  static inline constexpr std::size_t max_weight{3};

  using allocator_type = Alloc;

  using points_t = std::vector<point>;

  using points_alloc_t =
      all::rebind_alloc_t<allocator_type, std::pair<code const, points_t>>;

  using points_store = std::unordered_map<code,
                                          points_t,
                                          code_hash,
                                          std::equal_to<code>,
                                          points_alloc_t>;

  using count_store = std::array<std::size_t, max_weight>;

public:
  inline region(allocator_type const& alloc)
      : points_{points_alloc_t{alloc}} {
  }

  inline region()
      : region(allocator_type{}) {
  }

  inline void add(code const& key, point const& pt) {
    points_[key].push_back(pt);
    ++weight_count_[static_cast<std::size_t>(weight(key))];
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
  count_store weight_count_{};
};

template<std::size_t Width, std::size_t Height, typename Alloc>
class grid {
  static_assert(Width > 0);
  static_assert(Height > 0);

public:
  static inline constexpr std::size_t width{Width};
  static inline constexpr std::size_t height{Height};
  static inline constexpr std::size_t region_count{width * height};

  using allocator_type = Alloc;
  using region_type = region<allocator_type>;

  using regions_t = std::span<region_type const, region_count>;

private:
  template<std::size_t... Idxs>
  inline explicit grid(allocator_type const& alloc,
                       std::integer_sequence<std::size_t, Idxs...> /* unused */)
      : regions_{(Idxs, region_type{alloc})...} {
  }

public:
  inline explicit grid(allocator_type const& alloc)
      : grid(alloc, std::make_index_sequence<region_count>{}) {
  }

  inline grid()
      : grid(allocator_type{}) {
  }

  template<std::size_t... Idxs>
  inline void add(code const& key,
                  point pt,
                  std::index_sequence<Idxs...> /* unused */) {
    add_intern(key, pt, std::integral_constant<std::size_t, Idxs>{}...);
  }

  [[nodiscard]] inline regions_t regions() const noexcept {
    return regions_;
  }

  inline region_type& operator[](std::size_t index) noexcept {
    return regions_[index];
  }

  inline region_type const& operator[](std::size_t index) const noexcept {
    return regions_[index];
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
    regions_[Idx].add(key, pt);
  }

private:
  region_type regions_[region_count];
};

template<typename Ty>
concept gridlike = requires(Ty g) {
  typename Ty::allocator_type;

  requires std::unsigned_integral<decltype(Ty::width)>;
  requires Ty::width > 0;

  requires std::unsigned_integral<decltype(Ty::height)>;
  requires Ty::height > 0;

  {g.add(std::declval<code&&>(),
         std::declval<point&&>(),
         std::index_sequence<0>{})};
};

} // namespace kpr
