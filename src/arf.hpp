
// artifact filtering

#pragma once

#include "fgm.hpp"

#include <utility>

namespace arf {
namespace details {
  using unit_t = std::size_t;

  template<std::uint8_t Size>
  concept odd_size = Size % 2 == 1;

  template<std::uint8_t Size, std::uint8_t Idx>
  struct shifter {
    inline static void f(std::array<unit_t, Size>& arr) noexcept {
      constexpr auto idx{Size - Idx - 1};

      arr[idx + 1] |= (arr[idx] & 0xf) << (sizeof(unit_t) * 8 - 4);
      arr[idx] >>= 4;
    }
  };

  template<std::uint8_t Size>
  struct shifter<Size, 0> {
    inline static void f(std::array<unit_t, Size>& arr) noexcept {
      arr[Size - 1] >>= 4;
    }
  };

  template<std::uint8_t Size, std::size_t... Idxs>
  inline void shift(std::array<unit_t, Size>& arr,
                    std::index_sequence<Idxs...> /*unused*/) noexcept {
    ((shifter<Size, Idxs>::f(arr)), ...);
  }

  template<std::uint8_t Size>
  inline void shift(std::array<unit_t, Size>& arr) noexcept {
    shift(arr, std::make_index_sequence<Size>{});
  }

  template<std::uint8_t Size>
  requires odd_size<Size>
  class buffer {
  public:
    static constexpr std::uint8_t pixels_count{Size};
    static constexpr std::uint8_t unit_pixels{sizeof(unit_t) * 2};
    static constexpr std::uint8_t units_count{
        pixels_count / unit_pixels + (pixels_count % unit_pixels != 0 ? 1 : 0)};

    static constexpr std::uint8_t head_bit{pixels_count % unit_pixels * 4 - 4};

    using store_t = std::array<unit_t, units_count>;

  public:
    inline void push(cpl::nat_cc pixel) noexcept {
      shift(store_);
      store_[0] |= static_cast<unit_t>(value(pixel)) << head_bit;
      ++count_;
    }

    inline void reset() noexcept {
      count_ = 0;
    }

    [[nodiscard]] inline store_t const& data() const noexcept {
      return store_;
    }

    [[nodiscard]] inline bool ready() const noexcept {
      return count_ >= pixels_count;
    }

  private:
    store_t store_;
    std::uint32_t count_;
  };

  [[nodiscard]] inline std::size_t hash_impl(std::size_t value) noexcept {
    return value;
  }

  template<typename Ty, typename... Tys>
  [[nodiscard]] inline std::size_t
      hash_impl(std::size_t value, Ty data, Tys... others) noexcept {

    value ^= static_cast<std::size_t>(data);
    value *= 16777619U;

    return hash_impl(value, others...);
  }

  template<std::uint8_t Size, std::size_t... Idxs>
  [[nodiscard]] inline std::size_t
      hash_impl(std::array<unit_t, Size> const& arr,
                std::index_sequence<Idxs...> /*unused*/) noexcept {
    return hash_impl(2166136261U, arr[Idxs]...);
  }

  template<std::uint8_t Size>
  requires odd_size<Size>
  struct buffer_hash {
    [[nodiscar]] inline std::size_t
        operator()(buffer<Size> const& buffer) const noexcept {
      return hash_impl(buffer.data(), std::make_index_sequence<Size>{});
    }
  };

} // namespace details

template<std::uint8_t Size>
requires details::odd_size<Size>
void remove(mrl::matrix<cpl::nat_cc> const& image) {
  details::buffer<Size> buffer{};
  for (auto first{image.data()}, end{image.end()}; first < end;) {
    for (auto last{first + image.width()}; first < last; ++first) {
    }
  }
}
} // namespace arf
