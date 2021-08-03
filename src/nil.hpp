
// native image library

#include "sid.hpp"

#include "pngu.hpp"

#include <filesystem>
#include <fstream>

namespace nil {

template<typename Alloc>
[[nodiscard]] auto read_raw(std::filesystem::path const& filename,
                            mrl::dimensions_t const& dimension,
                            Alloc const& alloc) {
  sid::nat::aimg_t<Alloc> result{dimension, alloc};

  std::ifstream input{filename, std::ios::in | std::ios::binary};
  if (!input.is_open()) {
    return result;
  }

  input.read(reinterpret_cast<char*>(result.data()), dimension.area());

  return result;
}

[[nodiscard]] inline auto read_raw(std::filesystem::path const& filename,
                                   mrl::dimensions_t const& dimension) {
  return read_raw(filename, dimension, std::allocator<cpl::nat_cc>{});
}

namespace details {
  template<typename Pixel, typename Alloc, typename Cvt>
  inline void write_png(std::filesystem::path const& filename,
                        mrl::matrix<Pixel, Alloc> const& image,
                        Cvt const& cvt) {
    auto rgb{image.map(cvt)};
    png::write(filename, rgb.width(), rgb.height(), rgb.data());
  }
} // namespace details

template<typename Alloc>
inline void write_png(std::filesystem::path const& filename,
                      sid::nat::aimg_t<Alloc> const& image) {
  details::write_png(filename, image, [](cpl::nat_cc c) noexcept {
    return native_to_blend(c);
  });
}

} // namespace nil
