
// artifact removal filter

#pragma once

#include "fgm.hpp"

#include <intrin.h>

#include <array>
#include <memory>
#include <numeric>

namespace arf {
namespace details {
  namespace prob {
    template<std::uint8_t Depth>
    struct singular {
      static constexpr auto depth{Depth};
      static constexpr auto rep_size{depth / 8};

      [[nodiscard]] inline float const* values() const noexcept {
        return reinterpret_cast<float const*>(&rep_);
      }

      __m256 rep_[rep_size];
    };

    template<std::uint8_t Depth>
    struct composite {
      static constexpr auto depth{Depth};

      using singular_t = singular<depth>;
      using array_t = std::array<singular_t, depth>;

      array_t probabilities_;
    };
  } // namespace prob

  template<std::uint8_t Depth, std::uint8_t Width>
  struct kernel {
    static constexpr auto depth{Depth};
    static constexpr auto width{Width};

    using composite_t = prob::composite<depth>;
    using array_t = std::array<composite_t, width>;

    array_t rows_;
  };

  template<std::uint8_t Depth, typename Matrix>
  void calculate_intentsity(Matrix const& dots, float* output) noexcept {
    for (auto first{dots.data()}, last{dots.end()}; first < last;
         ++first, output += Depth) {
      auto total{
          std::accumulate(first->begin(), first->end(), std::uint32_t{})};
      if (total == 0) {
        std::fill_n(output, Depth, 0.0f);
      }
      else {
        for (std::uint8_t i{0}; i < Depth; ++i) {
          output[i] = static_cast<float>((*first)[i]) / total;
        }
      }
    }
  }

  template<std::uint8_t Kernel, std::uint8_t Depth>
  void kernel_pixel(kernel<Depth, Kernel>& output,
                    kernel<Depth, Kernel>& temp,
                    std::uint64_t norm,
                    float const* pos,
                    float const* window) noexcept {
    for (std::uint8_t ki{0}; ki < Kernel; ++ki) {
      for (std::uint8_t ci{0}; ci < Depth; ++ci, ++window) {
        if (auto pixel{*window}; pixel > 0) {
          auto scale{_mm256_set1_ps(pixel / norm)};

          auto c{temp.rows_[ki].probabilities_[ci].rep_};

          auto p{reinterpret_cast<__m256 const*>(pos)};
          auto y0{_mm256_sub_ps(_mm256_mul_ps(p[0], scale), c[0])};
          auto y1{_mm256_sub_ps(_mm256_mul_ps(p[1], scale), c[1])};

          auto s{output.rows_[ki].probabilities_[ci].rep_};

          auto t0{_mm256_add_ps(s[0], y0)};
          auto t1{_mm256_add_ps(s[1], y1)};

          c[0] = _mm256_sub_ps(_mm256_sub_ps(t0, s[0]), y0);
          c[1] = _mm256_sub_ps(_mm256_sub_ps(t1, s[1]), y1);

          s[0] = t0;
          s[1] = t1;
        }
      }
    }
  }

  template<std::uint8_t Kernel, std::uint8_t Depth>
  void generate_kernel(float const* input,
                       mrl::dimensions_t const& dim,
                       kernel<Depth, Kernel>& output) noexcept {
    constexpr auto ksize{static_cast<uint32_t>(Kernel) * Depth};
    constexpr auto khalf{static_cast<uint32_t>(Kernel / 2) * Depth};

    std::uint64_t isize{dim.area() * Depth};
    std::uint64_t norm{isize * Kernel};
    uint64_t rsize{dim.width_ * Depth - ksize};

    kernel<Depth, Kernel> temp{};
    for (auto end{input + isize}; input < end; input += ksize) {
      for (auto last{input + rsize}; input < last; input += Depth) {
        kernel_pixel(output, temp, norm, input + khalf, input);
      }
    }
  }

  template<std::uint8_t Kernel, std::uint8_t Depth>
  [[nodiscard]] auto output_pixel(kernel<Depth, Kernel> const& kernel,
                                  float const* window) noexcept {
    __m256 s[2]{};

    for (std::uint8_t ki{0}; ki < Kernel; ++ki) {
      for (std::uint8_t ci{0}; ci < Depth; ++ci, ++window) {
        if (auto pixel{*window}; pixel > 0) {
          auto scale{_mm256_set1_ps(pixel)};
          auto p{kernel.rows_[ki].probabilities_[ci].rep_};

          s[0] = _mm256_add_ps(s[0], _mm256_mul_ps(p[0], scale));
          s[1] = _mm256_add_ps(s[1], _mm256_mul_ps(p[1], scale));
        }
      }
    }

    auto v{reinterpret_cast<float const*>(s)};
    auto m{std::max_element(v, v + 16)};

    return static_cast<std::uint8_t>(m - v);
  }

  template<std::uint8_t Kernel, std::uint8_t Depth>
  [[nodiscard]] mrl::matrix<cpl::nat_cc>
      filter(float const* input,
             mrl::dimensions_t const& dim,
             kernel<Depth, Kernel> const& kernel) noexcept {
    constexpr auto ksize{static_cast<uint32_t>(Kernel) * Depth};

    std::uint64_t isize{dim.area() * Depth};
    std::uint64_t norm{isize * Kernel};
    uint64_t rsize{dim.width_ * Depth - ksize};

    mrl::matrix<cpl::nat_cc> result{dim};

    auto output{result.data()};
    for (auto end{input + isize}; input < end;
         input += ksize, output += Kernel) {
      for (auto last{input + rsize}; input < last; input += Depth, ++output) {
        output[Kernel / 2] = {output_pixel(kernel, input)};
      }
    }

    return result;
  }

} // namespace details

template<std::uint8_t Kernel, cpl::pixel Pixel, std::uint8_t Depth>
[[nodiscard]] mrl::matrix<cpl::nat_cc>
    filter(fgm::fragment<Depth, Pixel> const& fragment) {
  auto& dots{fragment.dots()};

  auto intentsity{std::make_unique<float[]>(Depth * dots.size())};
  details::calculate_intentsity<Depth>(dots, intentsity.get());

  details::kernel<Depth, Kernel> k{};
  details::generate_kernel<Kernel>(intentsity.get(), dots.dimensions(), k);
  return details::filter<Kernel>(intentsity.get(), dots.dimensions(), k);
}

} // namespace arf
