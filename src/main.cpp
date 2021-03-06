
#include "cte.hpp"
#include "fgc.hpp"
#include "fgm.hpp"
#include "kpe.hpp"
#include "kpm.hpp"
#include "mod.hpp"

#include "pngu.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <cstring>
#include <queue>
#include <string>

inline constexpr std::size_t screen_width = 388;
inline constexpr std::size_t screen_height = 312;

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

mrl::matrix<cpl::nat_cc> read_raw(std::string filename) {
  mrl::matrix<cpl::nat_cc> temp{screen_width, screen_height};

  std::ifstream input;
  input.open(ddir / filename, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return temp;
  }

  input.read(reinterpret_cast<char*>(temp.data()),
             static_cast<std::size_t>(temp.width()) * temp.height());
  input.close();

  return temp.crop(31, 53, 55, 105);
}

void write_rgb(std::string filename, mrl::matrix<cpl::rgb_bc> const& image) {
  png::write(ddir / filename, image.width(), image.height(), image.data());
}

class file_feed {
public:
  using image_type = fgc::collector::image_type;
  using allocator_type = image_type::allocator_type;

private:
  using vector_type = std::vector<std::filesystem::path>;

public:
  explicit file_feed(std::filesystem::path const& root) {
    using namespace std::filesystem;
    std::copy(
        directory_iterator(root), directory_iterator(), back_inserter(files_));
    std::sort(files_.begin(), files_.end(), [](auto& a, auto& b) {
      return std::stoi(a.filename().string()) <
             std::stoi(b.filename().string());
    });

    files_.resize(100);

    next_ = files_.begin();
  }

  [[nodiscard]] inline bool has_more() const noexcept {
    return next_ != files_.end();
  }

  [[nodiscard]] inline image_type produce(allocator_type alloc) {
    auto current{next_++};

    mrl::matrix<cpl::nat_cc, allocator_type> temp{
        screen_width, screen_height, alloc};

    std::ifstream input;
    input.open(*current, std::ios::in | std::ios::binary);
    if (!input.is_open()) {
      return temp;
    }

    input.read(reinterpret_cast<char*>(temp.data()),
               static_cast<std::size_t>(temp.width()) * temp.height());
    input.close();

    return temp.crop(31, 53, 55, 105);
  }

private:
  vector_type files_;
  vector_type::iterator next_;
};

int main() {
  constexpr std::size_t perf_loops{1000};

  auto image1{read_raw("raw1")};
  auto image2{read_raw("raw2")};

  using kpe_t = kpe::extractor<kpr::grid<4, 2, std::allocator<char>>, 16>;

  kpe_t extractor1{image1.width(), image1.height()};
  kpe_t extractor2{image1.width(), image1.height()};

  mrl::matrix<cpl::nat_cc> median1{image1.width(), image1.height()};
  mrl::matrix<cpl::nat_cc> median2{image1.width(), image1.height()};

  perf_test(
      [&image1, &median1, &extractor1] {
        [[maybe_unused]] auto grid{extractor1.extract(image1, median1)};
      },
      perf_loops,
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
      perf_loops,
      "match",
      false);

  cte::extractor<cpl::nat_cc>::allocator_type alloc{};
  cte::extractor<cpl::nat_cc> cext1{image1.width(), image1.height(), alloc};
  cte::extractor<cpl::nat_cc> cext2{image1.width(), image1.height(), alloc};

  perf_test([&cext1, &median1]() { auto contours{cext1.extract(median1)}; },
            perf_loops,
            "contour",
            false);

  mrl::matrix<cpl::nat_cc> recovered{image1.width(), image1.height()};

  auto contours1{cext1.extract(median1)};
  auto contours2{cext2.extract(median2)};

  mod::detector<cpl::nat_cc> mdet{2, 11};
  auto motion{
      mdet.detect(cext1.outline(), cext2.outline(), *offset, contours2)};

  mrl::matrix<cpl::rgb_bc> rgb_mh{image1.width(), image1.height()};
  mrl::matrix<cpl::rgb_bc> rgb_mv{image1.width(), image1.height()};

  perf_test(
      [&mdet, &cext1, &cext2, &offset, &contours2]() {
        auto motion{
            mdet.detect(cext1.outline(), cext2.outline(), *offset, contours2)};
      },
      perf_loops,
      "motion",
      false);

  for (auto const& contour : contours2) {
    auto a{motion[contour.id()]};
    auto [x, y]{motion[contour.id()]};

    if (x < 0) {
      contour.recover(rgb_mh.data(),
                      cpl::pack_to_blend(
                          {static_cast<std::uint8_t>(255 / 5 * -x)}, {}, {}));
    }
    else if (x > 0) {
      contour.recover(
          rgb_mh.data(),
          cpl::pack_to_blend({}, {static_cast<std::uint8_t>(255 / 5 * x)}, {}));
    }

    if (y < 0) {
      contour.recover(rgb_mv.data(),
                      cpl::pack_to_blend(
                          {static_cast<std::uint8_t>(255 / 5 * (-y))}, {}, {}));
    }
    else if (y > 0) {
      contour.recover(
          rgb_mv.data(),
          cpl::pack_to_blend({}, {static_cast<std::uint8_t>(255 / 5 * y)}, {}));
    }
  }

  mrl::matrix<cpl::nat_cc> diff{image1.width(), image1.height()};
  for (auto& region : grid1.regions()) {
    for (auto& [key, points] : region.points()) {
      cpl::nat_cc value{static_cast<std::uint8_t>(kpr::weight(key))};

      for (auto& [x, y] : points) {
        diff.data()[y * image1.width() + x] = value;
      }
    }
  }

  fgm::fragment<16, cpl::nat_cc> frag{image1.width(), image1.height()};

  frag.blit(0, 0, image1);
  frag.blit(std::get<0>(*offset), std::get<1>(*offset), image2);

  auto merged{frag.generate()};

  auto rgb_o1 = image1.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_o2 = image2.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_m1 = median1.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_m2 = median2.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_d = diff.map([](auto c) noexcept { return native_to_blend(c); });
  auto rgb_c =
      recovered.map([](auto c) noexcept { return native_to_blend(c); });

  auto rgb_g = merged.map([](auto c) noexcept { return native_to_blend(c); });

  write_rgb("original1.png", rgb_o1);
  write_rgb("original2.png", rgb_o2);
  write_rgb("median1.png", rgb_m1);
  write_rgb("median2.png", rgb_m2);
  write_rgb("diff.png", rgb_d);
  write_rgb("contours.png", rgb_c);

  write_rgb("motion_h.png", rgb_mh);
  write_rgb("motion_v.png", rgb_mv);

  write_rgb("merged.png", rgb_g);

  fgc::collector collector{image1.width(), image1.height()};
  collector.collect(file_feed{ddir / "seq"});
  auto map{collector.current().generate()};

  auto rgb_mp = map.map([](auto c) noexcept { return native_to_blend(c); });
  write_rgb("map.png", rgb_mp);

  return 0;
}
