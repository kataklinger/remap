
// artifact filtering

#pragma once

#include "fgm.hpp"

#include <intrin.h>
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
  requires odd_size<Size>
  class buffer {
  public:
    static constexpr std::uint8_t pixels_count{Size};
    static constexpr std::uint8_t unit_pixels{sizeof(unit_t) * 2};
    static constexpr std::uint8_t units_count{
        pixels_count / unit_pixels + (pixels_count % unit_pixels != 0 ? 1 : 0)};

    static constexpr std::uint8_t head_bit{pixels_count % unit_pixels * 4 - 4};

    using data_t = std::array<unit_t, units_count>;

  public:
    inline void push(cpl::nat_cc pixel) noexcept {
      shift(data_);
      data_[0] |= static_cast<unit_t>(value(pixel)) << head_bit;
    }

    [[nodiscard]] inline data_t const& data() const noexcept {
      return data_;
    }

    friend auto operator<=>(buffer const&, buffer const&) = default;

  private:
    data_t data_;
  };

  template<std::uint8_t Size>
  requires odd_size<Size>
  class counted_buffer {
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
      return count_ >= Size;
    }

    [[nodiscard]] buffer_t const& get() const noexcept {
      return buffer_;
    }

  private:
    buffer_t buffer_{};
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
  requires odd_size<Size>
  struct buffer_hash {
    [[nodiscar]] inline std::size_t
        operator()(buffer<Size> const& buf) const noexcept {
      return hash_impl(buf.data(),
                       std::make_index_sequence<buffer<Size>::units_count>{});
    }
  };

  template<std::uint8_t Size>
  using pattern_counter =
      std::unordered_map<buffer<Size>, std::uint32_t, buffer_hash<Size>>;

  template<std::uint8_t Size>
  requires odd_size<Size>
  [[nodiscard]] mrl::matrix<std::uint32_t>
      generate_heatmap_comp(fgm::fragment_blend const& fragment,
                            mrl::size_type instep,
                            mrl::size_type outstep,
                            mrl::size_type size,
                            mrl::size_type stride) {
    mrl::matrix<std::uint32_t*> result{fragment.image_.dimensions()};
    auto output{result.data()};

    auto width{fragment.image_.width()};

    details::pattern_counter<Size> counters{};
    details::counted_buffer<Size> buffer{};

    for (auto first{fragment.image_.data()}, col{first}, end{first + size};
         col < end;
         col += outstep) {
      buffer.reset();

      for (auto current{col}, last{current + stride}; current < last;
           current += instep) {
        for (; current < last && !fragment.mask_[current - first];
             current += instep) {
          buffer.reset();
        }

        buffer.push(*current);
        if (buffer.ready()) {
          auto& count{counters[buffer.get()]};
          ++count;

          output[current - first - (Size / 2) * instep] = &count;
        }
      }
    }

    return result.map([](auto v) { return v ? *v : 0; });
  }

  mrl::matrix<std::uint32_t> combine(mrl::matrix<std::uint32_t> const& left,
                                     mrl::matrix<std::uint32_t> const& right) {
    mrl::matrix<std::uint32_t> result{left.dimensions()};

    auto out{result.data()};
    auto a{left.data()}, b{right.data()};

    constexpr auto step = sizeof(__m256i) / sizeof(std::uint32_t);
    for (auto last{out + result.size() - result.size() % step}; out < last;
         out += step, a += step, b += step) {
      *reinterpret_cast<__m256i*>(out) =
          _mm256_add_epi32(*reinterpret_cast<__m256i const*>(a),
                           *reinterpret_cast<__m256i const*>(b));
    }

    for (auto last{result.end()}; out < last; ++out, ++a, ++b) {
      *out = *a + *b;
    }

    return result;
  }

  template<std::uint8_t Size>
  requires odd_size<Size>
  [[nodiscard]] mrl::matrix<std::uint32_t>
      generate_heatmap(fgm::fragment_blend const& fragment) {

    auto& image{fragment.image_};

    auto hor{generate_heatmap_comp<Size>(
        fragment, 1, image.width(), image.size(), image.width())};

    auto ver{generate_heatmap_comp<Size>(
        fragment, image.width(), 1, image.width(), image.size())};

    return combine(hor, ver);
  }
} // namespace details

} // namespace arf
