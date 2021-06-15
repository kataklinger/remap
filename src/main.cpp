
#include "arf.hpp"
#include "aws.hpp"
#include "cte.hpp"
#include "fde.hpp"
#include "fdf.hpp"
#include "fgm.hpp"
#include "fgs.hpp"
#include "frc.hpp"
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
inline constexpr mrl::dimensions_t screen_dimensions{screen_width,
                                                     screen_height};

inline std::uint64_t now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template<typename Fn>
requires std::invocable<Fn>
void perf_test(Fn&& fn, std::size_t count, std::string const& name, bool run) {
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
  mrl::matrix<cpl::nat_cc> temp{screen_dimensions};

  std::ifstream input;
  input.open(ddir / filename, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return temp;
  }

  input.read(reinterpret_cast<char*>(temp.data()), temp.dimensions().area());
  input.close();

  return temp.crop({32, 56, 55, 106});
}

void write_rgb(std::string filename, mrl::matrix<cpl::rgb_bc> const& image) {
  png::write(ddir / filename, image.width(), image.height(), image.data());
}

template<typename Iter>
void write_fragments(std::filesystem::path dir, Iter first, Iter last) {
  for (auto i{0}; first != last; ++first, ++i) {
    std::fstream output;
    output.open(dir / std::to_string(i), std::ios::out | std::ios::binary);

    auto dim{first->dots().dimensions()};
    output.write(reinterpret_cast<char const*>(&dim), sizeof(dim));

    output.write(reinterpret_cast<char const*>(first->dots().data()),
                 first->dots().dimensions().area() * sizeof(fgm::dot_t<16>));

    auto zero{first->zero()};
    output.write(reinterpret_cast<char const*>(&zero), sizeof(zero));

    auto frames{first->frames().size()};
    output.write(reinterpret_cast<char const*>(&frames), sizeof(frames));

    for (auto& frame : first->frames()) {
      output.write(reinterpret_cast<char const*>(&frame), sizeof(frame));
    }

    output.close();
  }
}

auto read_fragments(std::filesystem::path dir) {
  using namespace std::filesystem;
  using fragment_t = fgm::fragment<16>;

  std::vector<std::filesystem::path> files;
  std::copy(
      directory_iterator(dir), directory_iterator(), back_inserter(files));
  std::sort(files.begin(), files.end(), [](auto& a, auto& b) {
    return std::stoi(a.filename().string()) < std::stoi(b.filename().string());
  });

  std::vector<fragment_t> result;
  for (auto& file : files) {
    std::ifstream input;
    input.open(file, std::ios::in | std::ios::binary);

    mrl::dimensions_t dim{};
    input.read(reinterpret_cast<char*>(&dim), sizeof(mrl::dimensions_t));

    fragment_t::matrix_type temp{dim};
    input.read(reinterpret_cast<char*>(temp.data()),
               dim.area() * sizeof(fgm::dot_t<16>));

    fgm::point_t zero{};
    input.read(reinterpret_cast<char*>(&zero), sizeof(zero));

    std::size_t count{};
    input.read(reinterpret_cast<char*>(&count), sizeof(count));

    std::vector<fgm::frame> frames{};
    for (std::size_t i{0}; i < count; ++i) {
      fgm::frame frame{};
      input.read(reinterpret_cast<char*>(&frame), sizeof(frame));

      frames.push_back(frame);
    }

    input.close();

    result.emplace_back(
        std::move(temp), mrl::dimensions_t{1, 1}, zero, std::move(frames));
  }

  return result;
}

template<typename Image>
class file_feed {
public:
  using image_type = Image;
  using frame_type = ifd::frame<image_type>;

  using allocator_type = typename image_type::allocator_type;

private:
  using vector_type = std::vector<std::filesystem::path>;

public:
  explicit file_feed(std::filesystem::path const& root,
                     std::optional<mrl::region_t> crop = {})
      : crop_{crop} {
    using namespace std::filesystem;
    std::copy(
        directory_iterator(root), directory_iterator(), back_inserter(files_));
    std::sort(files_.begin(), files_.end(), [](auto& a, auto& b) {
      return std::stoi(a.filename().string()) <
             std::stoi(b.filename().string());
    });

    // files_.resize(900);

    next_ = files_.begin();
  }

  [[nodiscard]] inline bool has_more() const noexcept {
    return next_ != files_.end();
  }

  [[nodiscard]] inline frame_type produce() {
    return produce(allocator_type{});
  }

  [[nodiscard]] frame_type produce(allocator_type alloc) {
    auto current{next_++};

    auto count = current - files_.begin();
    if (count % 100 == 0) {
      auto this_time_{now()};
      if (last_time_ > 0) {
        std::cout << "#" << count << " ~ " << this_time_ - last_time_ << " @ "
                  << this_time_ - start_time_ << std::endl;
      }
      else {
        start_time_ = this_time_;
      }

      last_time_ = this_time_;
    }

    mrl::matrix<cpl::nat_cc, allocator_type> temp{screen_dimensions, alloc};

    auto no{static_cast<std::size_t>(next_ - files_.begin())};

    std::ifstream input;
    input.open(*current, std::ios::in | std::ios::binary);
    if (!input.is_open()) {
      return {no, temp};
    }

    input.read(reinterpret_cast<char*>(temp.data()), temp.dimensions().area());
    input.close();

    if (crop_) {
      return {no, temp.crop(*crop_)};
    }

    return {no, temp};
  }

private:
  vector_type files_;
  vector_type::iterator next_;

  std::optional<mrl::region_t> crop_;

  std::uint64_t start_time_{0};
  std::uint64_t last_time_{0};
};

int main() {
  constexpr std::size_t perf_loops{1000};

  auto image1{read_raw("raw1")};
  auto image2{read_raw("raw2")};

  using kpe_t = kpe::extractor<kpr::grid<4, 2, std::allocator<char>>, 16>;

  kpe_t extractor1{image1.dimensions()};
  kpe_t extractor2{image1.dimensions()};

  mrl::matrix<cpl::nat_cc> median1{image1.dimensions()};
  mrl::matrix<cpl::nat_cc> median2{image1.dimensions()};

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
  cte::extractor<cpl::nat_cc> cext1{image1.dimensions(), alloc};
  cte::extractor<cpl::nat_cc> cext2{image1.dimensions(), alloc};

  perf_test([&cext1, &median1]() { auto contours{cext1.extract(median1)}; },
            perf_loops,
            "contour",
            false);

  mrl::matrix<cpl::nat_cc> recovered{image1.dimensions()};

  auto contours1{cext1.extract(median1)};
  auto contours2{cext2.extract(median2)};

  mod::detector<cpl::nat_cc> mdet{2, 11};
  auto motion{
      mdet.detect(cext1.outline(), cext2.outline(), *offset, contours2)};

  mrl::matrix<cpl::rgb_bc> rgb_mh{image1.dimensions()};
  mrl::matrix<cpl::rgb_bc> rgb_mv{image1.dimensions()};

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

  mrl::matrix<cpl::nat_cc> diff{image1.dimensions()};
  for (auto& region : grid1.regions()) {
    for (auto& [key, points] : region.points()) {
      cpl::nat_cc value{static_cast<std::uint8_t>(kpr::weight(key))};

      for (auto& [x, y] : points) {
        diff.data()[y * image1.width() + x] = value;
      }
    }
  }

  fgm::fragment<16> frag{image1.dimensions()};

  frag.blit({0, 0}, image1, 0);
  frag.blit(*offset, image2, 1);

  auto merged{frag.blend().image_};

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

  // auto active{aws::scan(file_feed<mrl::matrix<cpl::nat_cc>>{ddir / "seq"},
  //                      {screen_width, screen_height})};

  // if (!active) {
  //  return 0;
  //}

  // std::optional<aws::window_info> active{
  //    std::in_place,
  //    mrl::region_t{31, 55, 334, 207},
  //    mrl::dimensions_t{screen_width, screen_height}};

  // frc::collector collector{active->bounds().dimensions()};
  // collector.collect(
  //    file_feed<frc::collector::image_type>{ddir / "seq", active->margins()});

  // auto& fragments{collector.fragments()};
  // write_fragments(ddir / "fgm", fragments.begin(), fragments.end());

  // auto fragments1{read_fragments(ddir / "fgm")};

  // auto spliced{fgs::splice<frc::collector::color_depth>(fragments1.begin(),
  //                                                      fragments1.end())};

  // auto smap{spliced.front().blend().image_};
  // auto rgb_smp = smap.map([](auto c) noexcept { return native_to_blend(c);
  // }); write_rgb("smap.png", rgb_smp);

  // auto filtered{fdf::filter(
  //    spliced,
  //    file_feed<mrl::matrix<cpl::nat_cc>>{ddir / "seq", active->margins()})};

  // write_fragments(ddir / "filt", filtered.begin(), filtered.end());

  auto fragments3{read_fragments(ddir / "filt")};

  auto master{std::max_element(fragments3.begin(),
                               fragments3.end(),
                               [](auto& lhs, auto& rhs) {
                                 return lhs.dots().size() < rhs.dots().size();
                               })
                  ->blend()};

  auto rgb_fmp =
      master.image_.map([](auto c) noexcept { return native_to_blend(c); });

  write_rgb("fmap.png", rgb_fmp);

  auto heat{arf::details::generate_heatmap<15>(master)};

  auto hmax{*std::max_element(heat.data(), heat.end(), std::less<>{})};

  std::cout << hmax << std::endl;
  auto rgb_ht = heat.map([hmax](auto c) noexcept {
    return cpl::intensity_to_blend({1.0f / std::powf(c / 2.0f, 0.5)});
  });

  write_rgb("heat.png", rgb_ht);

  return 0;
}
