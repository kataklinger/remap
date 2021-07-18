
// image compression definitions

#pragma once

#include "sid.hpp"

#include <concepts>
#include <utility>
#include <vector>

namespace icd {
using compressed_t = std::vector<std::uint8_t>;

template<typename Ty, typename Alloc>
concept compressor = requires(Ty c) {
  { c(std::declval<sid::nat::aimg_t<Alloc>>()) } -> std::same_as<compressed_t>;
};

template<typename Ty, typename Alloc>
concept decompressor = requires(Ty c) {
  {
    c(std::declval<compressed_t>(), std::declval<mrl::dimensions_t>())
    } -> std::same_as<sid::nat::aimg_t<Alloc>>;
};
} // namespace icd
