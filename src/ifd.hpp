
// image feed definitions

#pragma once

#include "sid.hpp"

#include <concepts>
#include <cstddef>
#include <utility>

namespace ifd {

template<typename Image>
struct frame {
  std::size_t number_;
  Image image_;
};

template<typename Ty, typename Alloc>
concept feeder = requires(Ty a) {
  { a.has_more() }
  noexcept->std::same_as<bool>;

  {
    a.produce(std::declval<Alloc&&>())
    } -> std::same_as<frame<sid::nat::aimg_t<Alloc>>>;
};

} // namespace ifd
