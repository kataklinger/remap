
// artifact filtering

#pragma once

#include "fgm.hpp"

#include <intrin.h>
#include <numbers>
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
        for (; current < last && !value(fragment.mask_[current - first]);
             current += instep) {
          buffer.reset();
        }
        if (current < end) {
          buffer.push(*current);
          if (buffer.ready()) {
            auto& count{counters[buffer.get()]};
            ++count;

            output[current - first - (Size / 2) * instep] = &count;
          }
        }
      }
    }

    return result.map([](auto v) { return v ? *v : 0; });
  }

  [[nodiscard]] mrl::matrix<float>
      combine(mrl::matrix<std::uint32_t> const& left,
              mrl::matrix<std::uint32_t> const& right) {
    mrl::matrix<float> result{left.dimensions()};

    auto out{result.data()};
    auto a{left.data()}, b{right.data()};

    constexpr auto step = sizeof(__m256i) / sizeof(std::uint32_t);
    for (auto last{out + result.size() - result.size() % step}; out < last;
         out += step, a += step, b += step) {
      auto sum{_mm256_cvtepi32_ps(
          _mm256_add_epi32(*reinterpret_cast<__m256i const*>(a),
                           *reinterpret_cast<__m256i const*>(b)))};

      *reinterpret_cast<__m256*>(out) =
          _mm256_rsqrt_ps(_mm256_div_ps(sum, _mm256_set1_ps(2.0f)));
    }

    for (auto last{result.end()}; out < last; ++out, ++a, ++b) {
      *out = 1.0f / std::sqrt((*a + *b) / 2.0f);
    }

    return result;
  }

  template<std::uint8_t Size>
  requires odd_size<Size>
  [[nodiscard]] mrl::matrix<float>
      generate_heatmap(fgm::fragment_blend const& fragment) {

    auto& image{fragment.image_};

    auto hor{generate_heatmap_comp<Size>(
        fragment, 1, image.width(), image.size(), image.width())};

    auto ver{generate_heatmap_comp<Size>(
        fragment, image.width(), 1, image.width(), image.size())};

    return combine(hor, ver);
  }

  [[nodiscard]] mrl::matrix<float> gauss_kernel(float dev) {
    using namespace std::numbers;

    auto size{static_cast<mrl::size_type>(std::ceil(6.0f * dev)) | 1};
    auto half{size / 2};

    float d{2 * dev * dev};
    float a{1 / (pi_v<float> * d)};

    mrl::matrix<float> result{mrl::dimensions_t{size, size}, float{}};
    auto out{result.data()};

    for (mrl::size_type y{0}; y < result.height(); ++y) {
      auto dy{static_cast<float>(y) - half};
      for (mrl::size_type x{0}; x < result.width(); ++x, ++out) {
        auto dx{static_cast<float>(x) - half};

        *out = a * std::pow(e_v<float>, -(dy * dy + dx * dx) / d);
      }
    }

    return result;
  }

  [[nodiscard]] sid::nat::dimg_t blur(fgm::fragment::matrix_type const& dots,
                                      mrl::matrix<float> const& heatmap,
                                      float dev) {
    auto kernel{details::gauss_kernel(dev)};

    auto size{kernel.width()};
    auto margin{size / 2};

    auto width{dots.width()};
    auto vstride{margin * width};

    auto kdata{kernel.data()};

    auto input{dots.data()};
    auto cond{heatmap.data()};

    sid::nat::dimg_t result{heatmap.dimensions()};
    auto output{result.data()};

    for (auto outer{input + margin}, ocend{dots.end() - vstride - margin};
         outer < ocend;
         outer += size) {
      for (auto orend{outer + dots.width() - size}; outer < orend; ++outer) {
        if (cond[outer - input] > 0.25f) {
          auto k{kdata};
          std::array<float, 16> temp{};

          for (auto inner{outer - vstride - margin},
               icend{outer + vstride - margin};
               inner < icend;
               inner += width - size) {
            for (auto irend{inner + size}; inner < irend; ++inner, ++k) {
              for (std::uint8_t i{0}; i < 16; ++i) {
                if ((*outer)[i] > 0.0f) {
                  temp[i] += (*inner)[i] * *k;
                }
              }
            }
          }

          output[outer - input] = {static_cast<std::uint8_t>(
              std::max_element(temp.begin(), temp.end()) - temp.begin())};
        }
        else {
          output[outer - input] = {static_cast<std::uint8_t>(
              std::max_element(outer->begin(), outer->end()) - outer->begin())};
        }
      }
    }

    return result;
  }

} // namespace details

template<std::uint8_t Size>
using filter_size = std::integral_constant<std::uint8_t, Size>;

template<typename Callback, std::uint8_t Size>
[[nodiscard]] sid::nat::dimg_t
    filter(fgm::fragment const& fragment,
           Callback&& cb,
           float dev,
           std::integral_constant<std::uint8_t, Size> /*unused*/) {
  auto heatmap{details::generate_heatmap<Size>(fragment.blend())};
  auto result{details::blur(fragment.dots(), heatmap, dev)};

  cb(result, heatmap);

  return result;
}

} // namespace arf
