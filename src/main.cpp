
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

std::filesystem::path const ddir{"../../../data/"};

void read_raw(std::string filename, mrl::matrix<cpl::nat_cc>& output) {

  std::ifstream input;

  input.open(ddir / filename, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return;
  }

  input.read(reinterpret_cast<char*>(output.data()),
             output.width() * output.height());
  input.close();
}

int main() {
  mrl::matrix<cpl::nat_cc> image1{screen_width, screen_height};
  mrl::matrix<cpl::nat_cc> image2{screen_width, screen_height};

  read_raw("raw1", image1);
  read_raw("raw2", image2);

  extractor_t extractor1{screen_width, screen_height};
  extractor_t extractor2{screen_width, screen_height};

  mrl::matrix<cpl::nat_cc> median1{screen_width, screen_height};
  mrl::matrix<cpl::nat_cc> median2{screen_width, screen_height};

  perf_test(
      [&image1, &median1, &extractor1] {
        [[maybe_unused]] auto grid{extractor1.extract(image1, median1)};
      },
      100,
      "median",
      false);

  auto grid1{extractor1.extract(image1, median1)};
  auto grid2{extractor2.extract(image2, median2)};

  match_config cfg;
  auto offset{kpm::match(cfg, grid1, grid2)};

  perf_test(
      [&cfg, &grid1, &grid2]() {
        [[maybe_unused]] auto result{kpm::match(cfg, grid1, grid2)};
      },
      100,
      "match",
      false);

  cte::v1::extractor<cpl::nat_cc>::allocator_type alloc{};
  cte::v1::extractor<cpl::nat_cc> cext1{screen_width, screen_height, alloc};
  cte::v1::extractor<cpl::nat_cc> cext2{screen_width, screen_height, alloc};

  perf_test([&cext1, &median1]() { auto contours{cext1.extract(median1)}; },
            100,
            "contour",
            false);

  mrl::matrix<cpl::nat_cc> recovered{screen_width, screen_height};

  auto contours1{cext1.extract(median1)};
  auto contours2{cext2.extract(median2)};
  for (auto const& contour : contours1) {
    contour.recover(recovered.data(), std::true_type{});
  }

  mod::detector<cpl::nat_cc> mdet{1, 11};
  auto motion{
      mdet.detect(cext1.outline(), cext2.outline(), *offset, contours2)};

  mrl::matrix<cpl::nat_cc> diff{screen_width, screen_height};
  for (auto& region : grid1.regions()) {
    for (auto& [key, points] : region.points()) {
      cpl::nat_cc value{static_cast<std::uint8_t>(kpr::weight(key))};

      for (auto& [x, y] : points) {
        diff.data()[y * image1.width() + x] = value;
      }
    }
  }

  auto rgb_o1 = image1.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_o2 = image2.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_m = median1.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_d = diff.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_c =
      recovered.map([](auto c) noexcept { return native_to_blend(c); });

  png::write(
      ddir / "original1.png", screen_width, screen_height, rgb_o1.data());
  png::write(
      ddir / "original2.png", screen_width, screen_height, rgb_o2.data());
  png::write(ddir / "median1.png", screen_width, screen_height, rgb_m.data());
  png::write(ddir / "diff.png", screen_width, screen_height, rgb_d.data());
  png::write(ddir / "contours.png", screen_width, screen_height, rgb_c.data());

  return 0;
}
