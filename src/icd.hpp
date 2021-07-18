
// image compression definitions

#pragma once

#include "sid.hpp"

#include <concepts>
#include <utility>
#include <vector>

namespace icd {
using compressed = std::vector<std::uint8_t>;

template<typename Ty, typename Alloc>
concept compressor = requires(Ty c) {
  { c(std::declval<sid::nat::imgal_t<Alloc>>()) } -> std::same_as<compressed>;
};

template<typename Ty, typename Alloc>
concept decompressor = requires(Ty c) {
  {
    c(std::declval<compressed>(), std::declval<mrl::dimensions_t>())
    } -> std::same_as<sid::nat::imgal_t<Alloc>>;
};
} // namespace icd
