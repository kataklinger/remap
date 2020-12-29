
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace cpl {
template<typename Ty>
concept pixel_color = std::is_integral_v<Ty> || std::is_floating_point_v<Ty>;

struct rgb_component_color_t {};
struct rgb_blended_color_t {};

struct nat_coded_color_t {};
struct nat_ordered_value_t {};

struct grs_intensity_value_t {};

template<pixel_color Ty, typename Tag>
struct color_t {
  Ty value;
};

using rgb_cc = color_t<std::uint8_t, rgb_component_color_t>;
using rgb_bc = color_t<std::uint32_t, rgb_blended_color_t>;
using rgb_pc = std::tuple<rgb_cc, rgb_cc, rgb_cc>;

using nat_cc = color_t<std::uint8_t, nat_coded_color_t>;
using nat_ov = color_t<std::uint8_t, nat_ordered_value_t>;

using grs_iv = color_t<float, grs_intensity_value_t>;

inline constexpr rgb_bc native_to_blend_map[] = {{0x00000000},
                                                 {0x00FFFFFF},
                                                 {0x0068372B},
                                                 {0x0070A4B2},
                                                 {0x006F3D86},
                                                 {0x00588D43},
                                                 {0x00352879},
                                                 {0x00B8C76F},
                                                 {0x006F4F25},
                                                 {0x00433900},
                                                 {0x009A6759},
                                                 {0x00444444},
                                                 {0x006C6C6C},
                                                 {0x009AD284},
                                                 {0x006C5EB5},
                                                 {0x00959595}};

[[nodiscard]] inline constexpr rgb_bc native_to_blend(nat_cc color) noexcept {
  return native_to_blend_map[color.value];
}

[[nodiscard]] inline constexpr rgb_pc blend_to_pack(rgb_bc color) noexcept {
  return {{static_cast<std::uint8_t>(color.value)},
          {static_cast<std::uint8_t>(color.value >> 8)},
          {static_cast<std::uint8_t>(color.value >> 16)}};
}

[[nodiscard]] inline constexpr rgb_pc native_to_pack(nat_cc color) noexcept {
  return blend_to_pack(native_to_blend(color));
}

namespace details {
  [[nodiscard]] inline constexpr rgb_bc
      pack_to_blend(rgb_cc red, rgb_cc green, rgb_cc blue) noexcept {
    return {(static_cast<std::uint32_t>(red.value)) |
            (static_cast<std::uint32_t>(green.value) << 8) |
            (static_cast<std::uint32_t>(blue.value) << 16)};
  }

  [[nodiscard]] inline constexpr grs_iv
      pack_to_intensity(rgb_cc red, rgb_cc green, rgb_cc blue) noexcept {
    return {(0.3f * red.value + 0.59f * green.value + 0.11f * blue.value) /
            255.0f};
  }
} // namespace details

[[nodiscard]] inline constexpr rgb_bc
    pack_to_blend(rgb_cc red, rgb_cc green, rgb_cc blue) noexcept {
  return details::pack_to_blend(red, green, blue);
}

[[nodiscard]] inline constexpr rgb_bc pack_to_blend(rgb_pc color) noexcept {
  return std::apply(details::pack_to_blend, color);
}

[[nodiscard]] inline constexpr grs_iv
    pack_to_intentisy(rgb_cc red, rgb_cc green, rgb_cc blue) noexcept {
  return details::pack_to_intensity(red, green, blue);
}

[[nodiscard]] inline constexpr grs_iv pack_to_intensity(rgb_pc color) noexcept {
  return std::apply(details::pack_to_intensity, color);
}

[[nodiscard]] inline constexpr grs_iv
    blend_to_intensity(rgb_bc color) noexcept {
  return pack_to_intensity(blend_to_pack(color));
}

[[nodiscard]] inline constexpr grs_iv
    native_to_intensity(nat_cc color) noexcept {
  return blend_to_intensity(native_to_blend(color));
}

[[nodiscard]] inline constexpr rgb_pc
    intensity_to_pack(grs_iv intensity) noexcept {
  return {{static_cast<std::uint8_t>(255.0f * intensity.value)},
          {static_cast<std::uint8_t>(255.0f * intensity.value)},
          {static_cast<std::uint8_t>(255.0f * intensity.value)}};
}

[[nodiscard]] inline constexpr rgb_bc
    intensity_to_blend(grs_iv intensity) noexcept {
  return pack_to_blend(intensity_to_pack(intensity));
}

namespace details {
  [[nodiscard]] inline consteval std::array<nat_cc, 16>
      generate_ordered_to_native_map() noexcept {
    std::array<std::pair<std::uint8_t, grs_iv>, 16> buffer{};

    for (std::uint8_t i{0}; i < 16; ++i) {
      buffer[i] = std::pair{i, native_to_intensity({i})};
    }

    std::sort(begin(buffer), end(buffer), [](auto const& a, auto const& b) {
      return std::get<1>(a).value < std::get<1>(b).value;
    });

    std::array<nat_cc, 16> palette{};
    std::transform(begin(buffer),
                   end(buffer),
                   begin(palette),
                   [](auto const& e) { return nat_cc{std::get<0>(e)}; });

    return palette;
  }

  inline constexpr auto ordered_to_native_map =
      generate_ordered_to_native_map();

  [[nodiscard]] inline consteval std::array<nat_ov, 16>
      generate_native_to_ordered_map() noexcept {
    std::array<std::pair<std::uint8_t, std::uint8_t>, 16> buffer{};

    for (std::uint8_t i{0}; i < 16; ++i) {
      buffer[i] = std::pair{i, ordered_to_native_map[i].value};
    }

    std::sort(begin(buffer), end(buffer), [](auto const& a, auto const& b) {
      return std::get<1>(a) < std::get<1>(b);
    });

    std::array<nat_ov, 16> palette{};
    std::transform(begin(buffer),
                   end(buffer),
                   begin(palette),
                   [](auto const& e) { return nat_ov{std::get<0>(e)}; });

    return palette;
  }

  inline constexpr auto native_to_ordered_map =
      generate_native_to_ordered_map();
} // namespace details

[[nodiscard]] inline constexpr nat_ov native_to_ordered(nat_cc color) noexcept {
  return details::native_to_ordered_map[color.value];
}

[[nodiscard]] inline constexpr nat_cc ordered_to_native(nat_ov order) noexcept {
  return details::ordered_to_native_map[order.value];
}

[[nodiscard]] inline constexpr rgb_bc ordered_to_blend(nat_ov order) noexcept {
  return native_to_blend(ordered_to_native(order));
}

[[nodiscard]] inline constexpr grs_iv
    ordered_to_intensity(nat_ov order) noexcept {
  return native_to_intensity(ordered_to_native(order));
}

} // namespace cpl