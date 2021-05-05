
// artifact removal filter

#pragma once

#include "fgm.hpp"

#include <intrin.h>
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

    using composite_t = prob::composite_t<depth>;
    using array_t = std::array<composite_t, width>;

    array_t rows_;
  };

  template<typename Matrix, std::uint8_t Depth>
  void calculate_intentsity(Matrix const& dots, float* output) {
    for (auto first{dots.data()}, last{dots.end()}; first < last; ++first) {
      auto total{std::accumulate(dot.begin(), dot.end(), std::uint32_t{})};
      for (std::uint8_t i{0}; i < Depth; ++i) {
        output[0] = static_cast<float>(dots[i]) / total;
      }
    }
  }

  template<std::uint8_t Depth, std::uint8_t Kernel>
  void process_pixel(kernel<Depth, Kernel>& output,
                     kernel<Depth, Kernel>& temp,
                     std::uint64_t norm,
                     float const* window) {
    for (std::uint8_t ki{0}; ki < Kernel; ++ki) {
      for (std::uint8_t ci{0}; ci < Depth; ++ci, ++window) {
        if (auto pixel{*window}; pixel > 0) {
          auto scale{_mm256_set1_ps(pixel / norm)};

          auto c{temp.rows_[ki].probabilities_[ci].rep_};

          auto y0{_mm256_sub_ps(_mm256_mul_ps(*first, scale), c[0])};
          auto y1{_mm256_sub_ps(_mm256_mul_ps(*(first + 8), scale), c[1])};

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

  template<std::uint8_t Depth, std::uint8_t Kernel>
  void generate_kernel(float const* input,
                       mrl::size_type height,
                       mrl::size_type width,
                       kernel<Depth, Kernel>& output) {
    constexpr auto ksize{static_cast<uint32_t>(Depth) * Kernel};

    width *= Depth;

    std::uint64_t isize{height * width};
    std::uint64_t norm{isize * Kernel};
    uint64_t rsize{width - ksize};

    kernel<Depth, Kernel> temp{};
    for (auto end{input + isize}; input < end; input += ksize) {
      for (auto last{input + rsize}; input < last; input += Depth) {
        process_pixel(output, temp, norm, input);
      }
    }
  }

} // namespace details

template<cpl::pixel Pixel, std::uint8_t Depth, std::uint8_t Kernel>
[[nodiscard]] mrl::matrix<cpl::nat_cc>
    filter(fgm::fragment<Depth, Pixel> const& fragment) {

  auto& dots{fragment.dots()};

  auto intentsity{std::make_unique_for_overwrite<float>(Depth * dots.size())};
  calculate_intentsity(dots, intentsity.get());

  details::kernel<Depth, Kernel> kernel;
  details::generate_kernel(intentsity.get(), kernel);
}

} // namespace arf
