

#include "mpb.hpp"
#include "nic.hpp"

#include "pngu.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <cstring>
#include <queue>
#include <string>

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
                         std::size_t stagnation) const noexcept {
  }
};

struct frc_callback {
  inline void operator()(frc::fragment_t const& fragment,
                         frc::frame_type const& frame_type,
                         frc::image_type const& median,
                         frc::grid_type const& grid) const noexcept {
  }
};

struct fdf_callback {
  inline void operator()(fdf::fragment_t const& fragment,
                         std::size_t fragment_no,
                         sid::nat::dimg_t const& image,
                         std::size_t frame_no,
                         sid::nat::dimg_t const& median,
                         fgm::point_t const& pos,
                         fdf::contours_t const& foreground,
                         sid::mon::dimg_t const& mask) const noexcept {
  }
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

  inline void operator()(frc::fragment_list const& fragments) const noexcept {
  }

  inline void
      operator()(std::vector<frc::fragment_t> const& fragments) const noexcept {
  }
};

struct callbacks : aws_callback,
                   frc_callback,
                   fdf_callback,
                   arf_callback,
                   mpb_callbacks {};

class build_adapter {
public:
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

  [[nodiscard]] inline file_feed get_feed() const {
    return file_feed{screen_dimensions, files_};
  }

  [[nodiscard]] inline file_feed get_feed(mrl::region_t crop) const {
    return file_feed{screen_dimensions, files_, crop};
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

  [[nodiscard]] inline callbacks& get_callbacks() noexcept {
    return callbacks_;
  }

private:
  file_list files_;

  callbacks callbacks_{};
};

int main() {
  std::filesystem::path const ddir{"../../../data/"};

  mpb::builder builder{build_adapter{ddir / "seq"}};
  auto result{builder.build()};

  return 0;
}
