
// image feed definition

#pragma once

#include <concepts>
#include <cstddef>

namespace ifd {

template<typename Image>
struct frame {
  std::size_t number_;
  Image image_;
};

template<typename Ty>
concept feeder = requires(Ty a) {
  { a.has_more() }
  noexcept->std::same_as<bool>;
};
} // namespace ifd
