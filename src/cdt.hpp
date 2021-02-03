
#pragma once

#include <cstdint>
#include <tuple>

namespace cdt {
using offset_t = std::tuple<std::int32_t, std::int32_t>;

struct offset_hash {
  [[nodiscard]] std::size_t operator()(offset_t const& off) const noexcept {
    std::size_t hashed = 2166136261U;

    hashed ^= static_cast<std::size_t>(std::get<0>(off));
    hashed *= 16777619U;

    hashed ^= static_cast<std::size_t>(std::get<1>(off));
    hashed *= 16777619U;

    return hashed;
  }
};
} // namespace cdt
