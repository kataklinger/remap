
// keypoint extraction (v2)

#pragma once

#include <intrin.h>

#include "kpr.hpp"
#include "sid.hpp"

namespace kpe {
namespace v2 {

  using ksize_t = std::uint8_t;

  class extractor {
  public:
    static inline constexpr ksize_t kernel_size = 5;
    static inline constexpr ksize_t kernel_half = kernel_size / 2;

  public:
    extractor(mrl::size_type width, mrl::size_type height)
        : part_{width, height}
        , tote_{width, height} {
      unit_[0] = _mm_setr_epi8(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[1] = _mm_setr_epi8(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[2] = _mm_setr_epi8(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[3] = _mm_setr_epi8(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[4] = _mm_setr_epi8(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[5] = _mm_setr_epi8(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[6] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[7] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0);
      unit_[8] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
      unit_[9] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0);
      unit_[10] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
      unit_[11] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0);
      unit_[12] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
      unit_[13] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0);
      unit_[14] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0);
      unit_[15] = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
    }

    void extract(sid::nat::dimg_t const& image, sid::nat::dimg_t& median) {
      grid_.clear();

      auto tmp = part_.data();
      for (auto row{image.data()}, last{image.end()}; row < last;
           row += image.width()) {
        sum_row(row, tmp);
        ++tmp;
      }

      col_sum(image);
    }

    [[nodiscard]] kpr::grid const& grid() const noexcept {
      return grid_;
    }

  private:
    inline void sum_row(cpl::nat_cc const* row, __m256i* output) noexcept {
      auto last{row + part_.width()};

      auto pixel{cpl::native_to_ordered(*(row++))};
      auto buffer{push_pixel_buffer({}, pixel)};

      __m256i sum{get_unit_hi(pixel)};

      for (auto i{0}; i < 3; ++i) {
        pixel = native_to_ordered(*(row++));
        buffer = push_pixel_buffer(buffer, pixel);

        sum = _mm256_add_epi8(sum, get_unit(pixel, pixel));
      }

      pixel = native_to_ordered(*(row++));
      buffer = push_pixel_buffer(buffer, pixel);

      sum = _mm256_add_epi8(sum, get_unit_hi(pixel));

      output += part_.height() * kernel_half;
      *output = sum;

      while (row < last) {
        pixel = native_to_ordered(*(row++));

        sum = _mm256_add_epi8(
            _mm256_sub_epi8(sum,
                            get_unit({(buffer >> 4) & 0xf}, {buffer & 0xf})),
            get_unit({(buffer >> (4 * (kernel_size - 1))) & 0xf}, pixel));

        output += part_.height();
        *output = sum;

        buffer = push_pixel_buffer(buffer, pixel);
      }
    }

    inline [[nodiscard]] std::uint64_t
        push_pixel_buffer(std::uint64_t buffer,
                          cpl::nat_ov pixel) const noexcept {
      return (buffer >> 4) | (static_cast<std::uint64_t>(pixel.value)
                              << (4 * (kernel_size - 1)));
    }

    void col_sum(sid::nat::dimg_t const& image) {
      auto start{image.data() + image.width() * kernel_half};
      auto out{tote_.data() + (tote_.width() + 1) * kernel_half};

      auto first{part_.data() + image.height() * kernel_half},
          last{part_.data() + image.height() * (image.width() - kernel_half)};

      for (; first < last; first += part_.height(), ++out) {
        auto col{first};
        auto row{out};

        __m256i sum5{*(col++)};
        __m256i sum3{*(col++)};

        sum3 = _mm256_add_epi8(sum3, *(col++));
        sum3 = _mm256_add_epi8(sum3, *(col++));
        sum5 = _mm256_add_epi8(sum5, *(col++));
        sum5 = _mm256_add_epi8(sum5, sum3);

        *row = _mm256_blend_epi32(sum3, sum5, 0xf0);

        auto first_in{col}, last_in{first + part_.height()};

        for (; first_in < last_in; ++first_in) {
          row += part_.width();

          auto temp{_mm256_sub_epi8(sum5, *(first_in - kernel_size))};
          sum3 = _mm256_sub_epi8(temp, *(first_in - (kernel_size - 1)));
          sum5 = _mm256_add_epi8(temp, *first_in);

          *row = _mm256_blend_epi32(sum3, sum5, 0xf0);
        }
      }
    }

    void med() {
    }

    void encode_keypoint(cpl::nat_cc const* img,
                         std::byte* buffer,
                         std::uint8_t weight) const noexcept {
      img -= (part_.width() + 1) * kernel_half;

      img = extract_even(img, buffer);
      img = extract_odd(img, buffer + 2);
      img = extract_even(img, buffer + 5);
      img = extract_odd(img, buffer + 7);
      img = extract_even(img, buffer + 10);
      buffer[12] |= static_cast<std::byte>(weight);
    }

    inline cpl::nat_cc const* extract_even(cpl::nat_cc const* img,
                                           std::byte* buffer) const noexcept {
      std::uint32_t a;
      std::memcpy(&a, img, sizeof(a));
      std::uint8_t b{img[4].value};

      buffer[0] = static_cast<std::byte>(a | (a >> 4));
      buffer[1] = static_cast<std::byte>((a >> 16) | (a >> 20));
      buffer[2] = static_cast<std::byte>(b << 4);

      return img + part_.width();
    }

    inline cpl::nat_cc const* extract_odd(cpl::nat_cc const* img,
                                          std::byte* buffer) const noexcept {
      std::uint32_t a;
      std::memcpy(&a, img + 1, sizeof(a));
      std::uint8_t b{img[0].value};

      buffer[0] |= static_cast<std::byte>(b);
      buffer[1] = static_cast<std::byte>(a | (a >> 4));
      buffer[2] = static_cast<std::byte>((a >> 16) | (a >> 20));

      return img + part_.width();
    }

    inline [[nodiscard]] __m256i get_unit(cpl::nat_ov low,
                                          cpl::nat_ov hi) const noexcept {
      return _mm256_inserti128_si256(
          _mm256_castsi128_si256(unit_[low.value]), unit_[hi.value], 1);
    }

    inline [[nodiscard]] __m256i get_unit_low(cpl::nat_ov low) const noexcept {
      return _mm256_castsi128_si256(unit_[low.value]);
    }

    inline [[nodiscard]] __m256i get_unit_hi(cpl::nat_ov hi) const noexcept {
      return _mm256_inserti128_si256(
          _mm256_castsi128_si256({}), unit_[hi.value], 1);
    }

  private:
    __m128i unit_[16];

    mrl::matrix<__m256i> part_;
    mrl::matrix<__m256i> tote_;

    kpr::grid grid_;
  };

} // namespace v2
} // namespace kpe
