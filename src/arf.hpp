
// artifact filtering

#pragma once

#include "fgm.hpp"

#include <unordered_map>
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
  requires odd_size<Size> class buffer {
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
    }

    [[nodiscard]] inline store_t const& data() const noexcept {
      return store_;
    }

    friend auto operator<=>(buffer const&, buffer const&) = default;

  private:
    store_t store_;
  };

  template<std::uint8_t Size>
  requires odd_size<Size> class counted_buffer {
  public:
    using buffer_t = buffer<Size>;

  public:
    inline void push(cpl::nat_cc pixel) noexcept {
      buffer_.push(pixel);
      ++count_;
    }

    inline void reset() noexcept {
      count_ = 0;
    }

    [[nodiscard]] inline bool ready() const noexcept {
      return count_ >= pixels_count;
    }

    [[nodiscard]] buffer_t const& get() const noexcept {
      return buffer_;
    }

  private:
    buffer_t buffer_;
    std::uint32_t count_{};
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
  requires odd_size<Size> struct buffer_hash {
    [[nodiscar]] inline std::size_t
        operator()(buffer<Size> const& buffer) const noexcept {
      return hash_impl(buffer.data(), std::make_index_sequence<Size>{});
    }
  };

  template<std::uint8_t Size>
  using pattern_counter =
      std::unordered_map<buffer<Size>, std::size_t, buffer_hash<Size>>;

  template<std::uint8_t Size>
  requires details::odd_size<Size> [[nodiscard]] mrl::matrix<std::size_t>
      generate_heatmap(fragment_blend const& fragment) {
    details::pattern_counter<Size> counters{};
    details::counted_buffer<Size> buffer{};

    mrl::matrix<std::size_t> result{fragment.image_.dimensions()};
    auto output{result.data()};

    auto width{fragment.image_.width()};

    for (auto first{fragment.image_.data()},
         current{first},
         end{fragment.image_.end()};
         current < end;) {
      buffer.reset();

      for (auto last{current + width}; current < last; ++current) {
        for (; !fragment.mask_[current - first] && current < last; ++current) {
          buffer.reset();
        }

        buffer.push(*current);
        if (buffer.ready()) {
          auto count{++counters[buffer.get()]};
          output[current - first - Size / 2] = count;
        }
      }
    }

    return result;
  }


} // namespace details

} // namespace arf
