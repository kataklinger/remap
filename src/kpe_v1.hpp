
#pragma once

#include <intrin.h>

#include "cpl.hpp"
#include "kpr.hpp"
#include "mrl.hpp"

namespace kpe {
namespace v1 {
  using ksize_t = std::uint8_t;

  class extractor {
  public:
    static inline constexpr ksize_t kernel_size = 5;
    static inline constexpr ksize_t kernel_half = kernel_size / 2;

  public:
    extractor(mrl::size_type width, mrl::size_type height)
        : temp_{width, height} {
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

    void extract(mrl::matrix<cpl::nat_cc> const& image,
                 mrl::matrix<cpl::nat_cc>& median) {
      grid_.clear();

      auto tmp = temp_.data();
      for (auto row{image.data()}, last{image.end()}; row < last;
           row += image.width()) {
        sum_row(row, tmp);
        ++tmp;
      }

      col_out(image, median);
    }

    [[nodiscard]] kpr::grid const& grid() const noexcept {
      return grid_;
    }

  private:
    inline void sum_row(cpl::nat_cc const* row, __m256i* output) noexcept {
      auto last{row + temp_.width()};

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

      output += temp_.height() * kernel_half;
      *output = sum;

      while (row < last) {
        pixel = native_to_ordered(*(row++));

        sum = _mm256_add_epi8(
            _mm256_sub_epi8(sum,
                            get_unit({(buffer >> 4) & 0xf}, {buffer & 0xf})),
            get_unit({(buffer >> (4 * (kernel_size - 1))) & 0xf}, pixel));

        output += temp_.height();
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

    void col_out(mrl::matrix<cpl::nat_cc> const& image,
                 mrl::matrix<cpl::nat_cc>& median) {
      auto start{image.data() + image.width() * kernel_half};
      auto raw{start + kernel_half};
      auto out{median.data() + (median.width() + 1) * kernel_half};

      auto exl_sect_size{(temp_.width() / 4 - 8) * image.height()};
      auto mid_sect_size{16 * image.height()};

      auto first{temp_.data() + image.height() * kernel_half},
          last{temp_.data() + exl_sect_size};

      auto next_sect = [&first, &last](std::size_t delta) {
        first = last;
        last = first + delta;
      };

      col_out_sect(start, raw, out, first, last, std::index_sequence<0>{});

      next_sect(mid_sect_size);
      col_out_sect(start, raw, out, first, last, std::index_sequence<0, 1>{});

      next_sect(exl_sect_size);
      col_out_sect(start, raw, out, first, last, std::index_sequence<1>{});

      next_sect(mid_sect_size);
      col_out_sect(start, raw, out, first, last, std::index_sequence<1, 2>{});

      next_sect(exl_sect_size);
      col_out_sect(start, raw, out, first, last, std::index_sequence<2>{});

      next_sect(mid_sect_size);
      col_out_sect(start, raw, out, first, last, std::index_sequence<2, 3>{});

      first = last;
      last = temp_.data() + image.height() * (image.width() - kernel_half);
      col_out_sect(start, raw, out, first, last, std::index_sequence<3>{});
    }

    template<typename Outer>
    void col_out_sect(cpl::nat_cc const* row,
                      cpl::nat_cc const*& raw,
                      cpl::nat_cc*& out,
                      __m256i const* first,
                      __m256i const* last,
                      Outer /*unused*/) {
      for (; first < last; first += temp_.height(), ++raw, ++out) {
        auto x = static_cast<mrl::size_type>(raw - row);
        col_in(x, raw, first, out, Outer{});
      }
    }

    template<typename Outer>
    void col_in(mrl::size_type x,
                cpl::nat_cc const* raw,
                __m256i const* col,
                cpl::nat_cc* out,
                Outer /*unused*/) {
      constexpr std::index_sequence<0> lseq{};
      constexpr std::index_sequence<0, 1> mseq{};
      constexpr std::index_sequence<1> hseq{};

      auto first{col};

      __m256i sum5{*(col++)};
      __m256i sum3{*(col++)};

      sum3 = _mm256_add_epi8(sum3, *(col++));
      sum3 = _mm256_add_epi8(sum3, *(col++));
      sum5 = _mm256_add_epi8(sum5, *(col++));
      sum5 = _mm256_add_epi8(sum5, sum3);

      [[unlikely]] if (auto weight{compute_pixel(*raw, sum3, sum5, out)};
                       weight != 0) {
        kpr::code code;
        encode_keypoint(raw, code.data(), weight);
        grid_.add(code,
                  kpr::point{x, kernel_half},
                  kpr::grid_explode<2, Outer, decltype(lseq)>{});
      }

      auto lend{first + (temp_.height() / 2) - 8};
      auto mend{lend + 16};
      auto hend{first + temp_.height()};

      col_in_sect(x, raw, first, out, sum3, sum5, col, lend, Outer{}, lseq);
      col_in_sect(x, raw, first, out, sum3, sum5, lend, mend, Outer{}, mseq);
      col_in_sect(x, raw, first, out, sum3, sum5, mend, hend, Outer{}, hseq);
    }

    template<typename Outer, typename Inner>
    void col_in_sect(mrl::size_type x,
                     cpl::nat_cc const*& raw,
                     __m256i const* col,
                     cpl::nat_cc*& out,
                     __m256i& sum3,
                     __m256i& sum5,
                     __m256i const* first,
                     __m256i const* last,
                     Outer /*unused*/,
                     Inner /*unused*/) {
      for (; first < last; ++first) {
        raw += temp_.width();
        out += temp_.width();

        auto temp{_mm256_sub_epi8(sum5, *(first - kernel_size))};
        sum3 = _mm256_sub_epi8(temp, *(first - (kernel_size - 1)));
        sum5 = _mm256_add_epi8(temp, *first);

        [[unlikely]] if (auto weight{compute_pixel(*raw, sum3, sum5, out)};
                         weight != 0) {
          auto y = static_cast<mrl::size_type>(first - col - kernel_half);

          kpr::code code;
          encode_keypoint(raw, code.data(), weight);
          grid_.add(
              code, kpr::point{x, y}, kpr::grid_explode<2, Outer, Inner>{});
        }
      }
    }

    [[nodiscard]] std::uint8_t compute_pixel(cpl::nat_cc pixel,
                                             __m256i sum3,
                                             __m256i sum5,
                                             cpl::nat_cc* out) const noexcept {
      auto p1{cpl::native_to_ordered(pixel)};
      auto p3{median_pixel(_mm256_castsi256_si128(sum3), 4)};
      *out = cpl::ordered_to_native(p3);

      [[unlikely]] if (p1.value != p3.value) {
        auto p5{median_pixel(_mm256_extracti128_si256(sum5, 1), 12)};
        [[unlikely]] if (p3.value != p5.value) {
          return p1.value != p5.value ? 2 : 1;
        }
      }

      return 0;
    }

    [[nodiscard]] cpl::nat_ov median_pixel(__m128i totals,
                                           ksize_t half) const noexcept {
      ksize_t histogram[16];
      std::memcpy(histogram, &totals, sizeof(totals));

      int i{sizeof(histogram) - 1};
      for (ksize_t total{0}; i >= 0; --i) {
        total += histogram[i];
        if (total >= half) {
          return {static_cast<std::uint8_t>(i)};
        }
      }

      return {0};
    }

    void encode_keypoint(cpl::nat_cc const* img,
                         std::byte* buffer,
                         std::uint8_t weight) const noexcept {
      img -= (temp_.width() + 1) * kernel_half;

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

      return img + temp_.width();
    }

    inline cpl::nat_cc const* extract_odd(cpl::nat_cc const* img,
                                          std::byte* buffer) const noexcept {
      std::uint32_t a;
      std::memcpy(&a, img + 1, sizeof(a));
      std::uint8_t b{img[0].value};

      buffer[0] |= static_cast<std::byte>(b);
      buffer[1] = static_cast<std::byte>(a | (a >> 4));
      buffer[2] = static_cast<std::byte>((a >> 16) | (a >> 20));

      return img + temp_.width();
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

    mrl::matrix<__m256i> temp_;
    kpr::grid grid_;
  };

} // namespace v1
} // namespace kpe
