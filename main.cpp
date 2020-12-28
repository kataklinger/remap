﻿
#include "kpe_v1.hpp"
#include "pngu.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

inline constexpr std::size_t screen_width = 388;
inline constexpr std::size_t screen_height = 312;

inline uint64_t now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void perf_test(kpe::v1::extractor& extractor,
               mrl::matrix<cpl::nat_cc> const& image,
               mrl::matrix<cpl::nat_cc>& median,
               std::size_t count) {
  auto start = now();
  for (auto i{count}; i > 0; --i) {
    extractor.extract(image, median);
  }
  auto end = now();

  std::cout << "count = " << count << " time = " << end - start << "\n";
}

int main() {
  std::filesystem::path ddir{"../../../data/"};

  std::ifstream input;

  input.open(ddir / "raw", std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return 0;
  }

  mrl::matrix<cpl::nat_cc> image{screen_width, screen_height};
  mrl::matrix<cpl::nat_cc> median{screen_width, screen_height};

  input.read(reinterpret_cast<char*>(image.data()),
             screen_width * screen_height);
  input.close();

  kpe::v1::extractor extractor{screen_width, screen_height};

  // perf_test(extractor, image, median, 100);

  extractor.extract(image, median);

  mrl::matrix<cpl::nat_cc> diff{screen_width, screen_height};

  for (auto const& section : extractor.grid().sections()) {
    for (auto const& keypoint : section) {
      auto const& [x, y] = keypoint.second;
      std::size_t offset{y * image.width() + x};

      diff.data()[offset] = {static_cast<std::uint8_t>(
          keypoint.first[12] & static_cast<std::byte>(0xf))};
    }
  }

  auto rgb_o = image.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_m = median.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_d = diff.map([](auto c) noexcept { return native_to_blend(c); });

  png::write(ddir / "original.png", screen_width, screen_height, rgb_o.data());
  png::write(ddir / "median.png", screen_width, screen_height, rgb_m.data());
  png::write(ddir / "diff.png", screen_width, screen_height, rgb_d.data());

  return 0;
}
