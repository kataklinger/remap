
#include "cte_v1.hpp"
#include "kpe_v1.hpp"
#include "kpm.hpp"
#include "mod.hpp"
#include "pngu.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <cstring>
#include <queue>

inline constexpr std::size_t screen_width = 388;
inline constexpr std::size_t screen_height = 312;

using extractor_t = kpe::v1::extractor<kpr::grid<4, 2>, 16>;

inline uint64_t now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template<typename Fn>
requires std::invocable<Fn> void
    perf_test(Fn&& fn, std::size_t count, std::string const& name, bool run) {
  if (!run) {
    return;
  }

  auto start = now();
  for (auto i{count}; i > 0; --i) {
    fn();
  }
  auto end = now();

  std::cout << "[ " << name << " ] "
            << "[ count: " << count << "; time: " << end - start << " ]\n";
}

struct match_config {
  using allocator_type = std::allocator<char>;
  static constexpr std::size_t weight_switch{10};
  static constexpr std::size_t region_votes{3};

  allocator_type get_allocator() const noexcept {
    return allocator_type{};
  }
};

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

  extractor_t extractor{screen_width, screen_height};

  perf_test(
      [&image, &median, &extractor] {
        [[maybe_unused]] auto grid{extractor.extract(image, median)};
      },
      100,
      "median",
      false);

  auto grid{extractor.extract(image, median)};

  match_config cfg;
  perf_test(
      [&cfg, &grid]() {
        [[maybe_unused]] auto result{kpm::match(cfg, grid, grid)};
      },
      100,
      "match",
      false);

  cte::v1::extractor<cpl::nat_cc>::allocator_type alloc{};
  cte::v1::extractor<cpl::nat_cc> cext{screen_width, screen_height, alloc};

  perf_test([&cext, &median]() { auto contours{cext.extract(median)}; },
            100,
            "contour",
            false);

  mrl::matrix<cpl::nat_cc> recovered{screen_width, screen_height};

  auto contours{cext.extract(median)};
  for (auto const& contour : contours) {
    contour.recover(recovered.data(), std::true_type{});
  }

  mod::detect(cext.outline(), cext.outline(), {});

  mrl::matrix<cpl::nat_cc> diff{screen_width, screen_height};
  for (auto& region : grid.regions()) {
    for (auto& [key, points] : region.points()) {
      cpl::nat_cc value{static_cast<std::uint8_t>(kpr::weight(key))};

      for (auto& [x, y] : points) {
        diff.data()[y * image.width() + x] = value;
      }
    }
  }

  auto rgb_o = image.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_m = median.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_d = diff.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_c =
      recovered.map([](auto c) noexcept { return native_to_blend(c); });

  png::write(ddir / "original.png", screen_width, screen_height, rgb_o.data());
  png::write(ddir / "median.png", screen_width, screen_height, rgb_m.data());
  png::write(ddir / "diff.png", screen_width, screen_height, rgb_d.data());
  png::write(ddir / "contours.png", screen_width, screen_height, rgb_c.data());

  return 0;
}
