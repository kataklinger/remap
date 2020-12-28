
#pragma once

#include "mrl.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace kpr {
using point = std::tuple<mrl::size_type, mrl::size_type>;

static inline constexpr uint8_t keycode_length = 13;
using code = std::array<std::byte, keycode_length>;

struct code_hash {
  std::size_t operator()(code const& key) const noexcept {
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

struct grid {
public:
  using section =
      std::unordered_multimap<code, point, code_hash, std::equal_to<code>>;

  static inline constexpr std::size_t section_count{8};

  using storage = std::array<section, section_count>;

public:
  template<std::size_t... Idxs>
  inline void
      add(code const& key, point point, std::index_sequence<Idxs...> /**/) {
    add_intern(key, point, std::integral_constant<std::size_t, Idxs>{}...);
  }

  inline void clear() noexcept {
    for (auto& section : sections_) {
      section.clear();
    }
  }

  inline storage const& sections() const noexcept {
    return sections_;
  }

private:
  template<typename... Idxs>
  inline void add_intern(code const& key, point point, Idxs... idxs) {
    ((( void )add_intern(key, point, idxs)), ...);
  }

  template<std::size_t Idx>
  inline void add_intern(code const& key,
                         point point,
                         std::integral_constant<std::size_t, Idx> /*unused*/) {
    sections_[Idx].insert({key, point});
  }

private:
  storage sections_;
};

} // namespace kpr
