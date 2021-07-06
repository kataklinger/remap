
// native-color c64 compression

#include "cpl.hpp"
#include "mrl.hpp"

namespace ncc {

template<typename Alloc>
[[nodiscard]] std::vector<std::uint8_t>
    compress(mrl::matrix<cpl::nat_cc, Alloc> const& image) {
  std::vector<std::uint8_t> result{};

  std::vector<std::uint8_t> buffer{};

  std::uint16_t seq_len{1}, rep_len{1};

  auto first{image.data()};
  std::uint8_t current{value(*(first++))};
  buffer.push_back(current << 4);

  auto write_buf = [&](std::uint16_t len) {
    if (len <= 64) {
      result.push_back(0x80 | len);
    }
    else {
      result.push_back(0xc0 | (len >> 8));
      result.push_back(static_cast<std::uint8_t>(len));
    }

    result.insert(result.end(), buffer.begin(), buffer.end());
    buffer.clear();
  };

  auto write_rep = [&](std::uint16_t len) {
    if (len <= 6) {
      result.push_back(((len - 3) << 4) | current);
    }
    else {
      auto bytes{len > 256 ? 2 : 1};
      result.push_back(0x40 | (bytes << 4) | current);
      result.push_back(static_cast<std::uint8_t>(len));
      if (bytes == 2) {
        result.push_back(static_cast<std::uint8_t>(len >> 8));
      }
    }
  };

  for (auto last{image.end()}; first < last; ++first) {
    auto pixel{value(*first)};

    ++seq_len;

    if (current == pixel) {
      ++rep_len;

      if (rep_len < 3) {
        if ((seq_len & 1) == 0) {
          buffer.back() |= pixel;
        }
        else {
          buffer.push_back(pixel << 4);
        }
      }
      else if (rep_len == 3) {
        buffer.pop_back();

        if ((seq_len & 1) == 0) {
          buffer.back() &= 0xf0;
        }

        if (!buffer.empty()) {
          write_buf(seq_len - 3);
        }

        seq_len = 3;
      }
    }
    else {
      if (rep_len > 2) {
        write_rep(rep_len);
        seq_len = 1;
        buffer.push_back(pixel << 4);
      }
      else {
        if ((seq_len & 1) == 0) {
          buffer.back() |= pixel;
        }
        else {
          buffer.push_back(pixel << 4);
        }
      }

      rep_len = 1;
      current = pixel;
    }
  }

  if (rep_len > 2) {
    write_rep(rep_len);
  }
  else if (!buffer.empty()) {
    write_buf(seq_len);
  }

  return result;
}

template<typename Alloc>
[[nodiscard]] mrl::matrix<cpl::nat_cc>
    decompress(std::vector<std::uint8_t, Alloc> const& pack,
               mrl::dimensions_t const& dim) {
  mrl::matrix<cpl::nat_cc> result{dim};

  auto out{result.data()};

  auto z{0};
  for (auto it{pack.begin()}; it != pack.end(); ++it, ++z) {
    auto value{*it};

    switch (value & 0xc0) {
    case 0x00:
      for (auto count{(value >> 4) + 3}, i{0}; i < count; ++i) {
        *(out++) = {static_cast<std::uint8_t>(value & 0x0f)};
      }
      break;

    case 0x40: {
      std::size_t size{0};
      for (auto count{(value >> 4) & 3}, i{0}; i < count; ++i) {
        size |= static_cast<std::size_t>(*++it) << (8 * i);
      }

      for (auto i{0}; i < size; ++i) {
        *(out++) = {static_cast<std::uint8_t>(value & 0x0f)};
      }
    } break;

    case 0x80: {
      auto pixels{value & 0x3f};
      auto bytes{(pixels >> 1) + (pixels & 1)};

      for (auto i{0}, j{0}; i < bytes; ++i, ++j) {
        auto pair{*++it};

        *(out++) = {static_cast<std::uint8_t>(pair >> 4)};
        if (++j < pixels) {
          *(out++) = {static_cast<std::uint8_t>(pair & 0x0f)};
        }
      }
    } break;

    case 0xc0: {
      auto pixels{((value & 0x3f) << 8) + *++it};
      auto bytes{(pixels >> 1) + (pixels & 1)};

      for (auto i{0}, j{0}; i < bytes; ++i, ++j) {
        auto pair{*++it};

        *(out++) = {static_cast<std::uint8_t>(pair >> 4)};
        if (++j < pixels) {
          *(out++) = {static_cast<std::uint8_t>(pair & 0x0f)};
        }
      }
    } break;
    }
  }

  return result;
}

} // namespace ncc
