

#include "mpb.hpp"
#include "nic.hpp"

#include "pngu.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

using file_list = std::vector<std::filesystem::path>;

class file_feed {
public:
  inline file_feed(mrl::dimensions_t const& dimensions,
                   file_list const& files,
                   std::optional<mrl::region_t> crop = {})
      : dimensions_{dimensions}
      , files_{files}
      , next_{files_.begin()}
      , crop_{crop} {
  }

  [[nodiscard]] inline bool has_more() const noexcept {
    return next_ != files_.end();
  }

  template<typename Alloc>
  [[nodiscard]] auto produce(Alloc alloc) {
    using image_type = sid::nat::aimg_t<Alloc>;
    using frame_type = ifd::frame<image_type>;

    image_type temp{dimensions_, alloc};

    std::ifstream input{*(next_++), std::ios::in | std::ios::binary};
    if (!input.is_open()) {
      return frame_type{frame_number(), temp};
    }

    input.read(reinterpret_cast<char*>(temp.data()), temp.dimensions().area());

    return crop_ ? frame_type{frame_number(), temp.crop(*crop_)}
                 : frame_type{frame_number(), temp};
  }

private:
  [[nodiscard]] inline std::size_t frame_number() const {
    return static_cast<std::size_t>(next_ - files_.begin() - 1);
  }

private:
  mrl::dimensions_t dimensions_;

  file_list files_;
  file_list::iterator next_;

  std::optional<mrl::region_t> crop_;
};

class perf_counter {
public:
  inline perf_counter(std::string name, std::size_t sample_size)
      : name_{name}
      , sample_size_{sample_size} {
  }

  bool count() {
    ++total_count_;
    ++sample_count_;

    if (sample_count_ == sample_size_) {
      auto current{now()};

      auto duration_sample{current - last_};
      auto duration_total{current - begin_};

      std::cout << "[" << name_ << " # " << std::setw(5) << total_count_
                << "] step avg: " << std::setw(4)
                << sample_count_ * 1000 / duration_sample
                << " fps; total avg: " << std::setw(4)
                << total_count_ * 1000 / duration_total
                << "fps; total:" << std::setw(5) << duration_total / 1000 << "s"
                << std::endl;

      sample_count_ = 0;
      last_ = current;

      return true;
    }

    if (total_count_ == 1) {
      begin_ = last_ = now();
    }

    return false;
  }

private:
  inline std::uint64_t now() const noexcept {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
        .count();
  }

private:
  std::string name_;
  std::size_t sample_size_;

  std::size_t sample_count_{};
  std::size_t total_count_{};

  std::uint64_t begin_{};
  std::uint64_t last_{};
};

class native_compression {
public:
  template<typename Alloc>
  [[nodiscard]] icd::compressed_t
      operator()(sid::nat::aimg_t<Alloc> const& image) const {
    return nic::compress(image);
  }

  [[nodiscard]] sid::nat::dimg_t
      operator()(icd::compressed_t const& compressed,
                 mrl::dimensions_t const& dim) const {
    return nic::decompress(compressed, dim);
  }
};

struct aws_callback {
  inline void operator()(aws::frame_type const& frame,
                         aws::heatmap_type const& heatmap,
                         aws::contour_type const& contour,
                         std::size_t stagnation) noexcept {
    counter_.count();
  }

private:
  perf_counter counter_{"aws", 100};
};

struct frc_callback {
  inline void operator()(fgm::fragment const& fragment,
                         frc::frame_type const& frame_type,
                         frc::image_type const& median,
                         frc::grid_type const& grid) noexcept {
    counter_.count();
  }

private:
  perf_counter counter_{"frc", 100};
};

struct fdf_callback {
  inline void operator()(fgm::fragment const& fragment,
                         std::size_t fragment_no,
                         sid::nat::dimg_t const& image,
                         std::size_t frame_no,
                         sid::nat::dimg_t const& median,
                         fgm::point_t const& pos,
                         fdf::contours_t const& foreground,
                         sid::mon::dimg_t const& mask) noexcept {
    counter_.count();
  }

private:
  perf_counter counter_{"fdf", 1000};
};

struct arf_callback {
  inline void operator()(sid::nat::dimg_t const& fragment,
                         mrl::matrix<float> const& heatmap) const noexcept {
  }
};

struct mpb_callbacks {
  inline void
      operator()(std::optional<aws::window_info> const& window) const noexcept {
  }

  inline void
      operator()(std::list<fgm::fragment> const& fragments) const noexcept {
  }

  inline void
      operator()(std::vector<fgm::fragment> const& fragments) const noexcept {
  }
};

struct callbacks : aws_callback,
                   frc_callback,
                   fdf_callback,
                   arf_callback,
                   mpb_callbacks {};

class build_adapter {
public:
  using callbacks_type = callbacks;
  using feed_type = file_feed;

  static constexpr mrl::dimensions_t screen_dimensions{388, 312};
  static constexpr float artifact_filter_dev{2.0f};
  using artifact_filter_size = arf::filter_size<15>;

public:
  inline explicit build_adapter(std::filesystem::path const& root) {
    using namespace std::filesystem;
    std::copy(directory_iterator{root},
              directory_iterator{},
              std::back_inserter(files_));

    std::sort(files_.begin(), files_.end(), [](auto& a, auto& b) {
      return stoi(a.filename().string()) < stoi(b.filename().string());
    });
  }

  [[nodiscard]] inline feed_type get_feed() const {
    return {screen_dimensions, files_};
  }

  [[nodiscard]] inline feed_type get_feed(mrl::region_t crop) const {
    return {screen_dimensions, files_, crop};
  }

  [[nodiscard]] inline native_compression get_compression() const {
    return native_compression{};
  }

  [[nodiscard]] inline mrl::dimensions_t
      get_screen_dimensions() const noexcept {
    return screen_dimensions;
  }

  [[nodiscard]] inline float get_artifact_filter_dev() const noexcept {
    return artifact_filter_dev;
  }

  [[nodiscard]] inline callbacks_type& get_callbacks() noexcept {
    return callbacks_;
  }

private:
  file_list files_;

  callbacks_type callbacks_{};
};

std::filesystem::path const ddir{"../../../data/"};

void write_rgb(std::string filename, mrl::matrix<cpl::rgb_bc> const& image) {
  png::write(ddir / filename, image.width(), image.height(), image.data());
}

int main() {
  mpb::builder builder{build_adapter{ddir / "seq"}};
  auto results{builder.build()};

  std::size_t i{0};
  for (auto& result : results) {
    auto map{result.map([](auto c) noexcept { return native_to_blend(c); })};
    write_rgb(std::format("art{}.png", i), map);
  }

  return 0;
}
