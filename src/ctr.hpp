
#pragma once

#include "cpl.hpp"

#include <optional>
#include <vector>

namespace ctr {
enum class edge_side : std::uint8_t {
  none = 0,
  left = 1,
  right = 2,
  top = 4,
  bottom = 8
};

namespace details {
  [[nodiscard]] inline bool test_side(edge_side tested,
                                      std::uint8_t desired) noexcept {
    return (static_cast<std::uint8_t>(tested) & desired) != 0;
  }

  [[nodiscard]] inline bool test_side(edge_side tested,
                                      edge_side desired) noexcept {
    return (static_cast<std::uint8_t>(tested) &
            static_cast<std::uint8_t>(desired)) != 0;
  }

} // namespace details

[[nodiscard]] inline bool is_left(edge_side side) noexcept {
  return details::test_side(side, edge_side::left);
}

[[nodiscard]] inline bool is_right(edge_side side) noexcept {
  return details::test_side(side, edge_side::right);
}

[[nodiscard]] inline bool is_horisontal(edge_side side) noexcept {
  return details::test_side(side,
                            static_cast<std::uint8_t>(edge_side::left) |
                                static_cast<std::uint8_t>(edge_side::right));
}

[[nodiscard]] inline bool is_top(edge_side side) noexcept {
  return details::test_side(side, edge_side::top);
}

[[nodiscard]] inline bool is_bottom(edge_side side) noexcept {
  return details::test_side(side, edge_side::bottom);
}

[[nodiscard]] inline bool is_vertical(edge_side side) noexcept {
  return details::test_side(side,
                            static_cast<std::uint8_t>(edge_side::top) |
                                static_cast<std::uint8_t>(edge_side::bottom));
}

[[nodiscard]] inline edge_side
    create_edge(bool left, bool right, bool top, bool bottom) {
  return static_cast<edge_side>(static_cast<std::uint8_t>(left) |
                                (static_cast<std::uint8_t>(right) << 1) |
                                (static_cast<std::uint8_t>(top) << 2) |
                                (static_cast<std::uint8_t>(bottom) << 3));
}

class edge {
public:
  inline edge(std::ptrdiff_t position, edge_side side) noexcept
      : rep_{static_cast<std::uint32_t>(position << 4) |
             static_cast<std::uint32_t>(side)} {
  }

  [[nodiscard]] inline std::uint32_t position() const noexcept {
    return rep_ >> 4;
  }

  [[nodiscard]] inline edge_side side() const noexcept {
    return static_cast<edge_side>(rep_ & 0xf);
  }

  friend auto operator<=>(edge, edge) = default;

private:
  std::uint32_t rep_;
};

namespace details {
  struct limits {
    inline limits() noexcept
        : lower_{std::numeric_limits<std::size_t>::max()}
        , upper_{0} {
    }

    inline limits(std::size_t lower, std::size_t upper) noexcept
        : lower_{lower}
        , upper_{upper} {
    }

    inline void update(std::size_t value) noexcept {
      if (value > upper_) {
        upper_ = value;
      }
      else if (value < lower_) {
        lower_ = value;
      }
    }

    [[nodiscard]] std::size_t size() const noexcept {
      return upper_ - lower_;
    }

    std::size_t lower_;
    std::size_t upper_;
  };
} // namespace details

class box {
public:
  const box(details::limits const& horizontal,
            details::limits const& vertical) noexcept
      : horizontal_(horizontal)
      , vertical_(vertical) {
  }

  [[nodiscard]] std::size_t left() const noexcept {
    return horizontal_.lower_;
  }

  [[nodiscard]] std::size_t right() const noexcept {
    return horizontal_.upper_;
  }

  [[nodiscard]] std::size_t width() const noexcept {
    return horizontal_.size();
  }

  [[nodiscard]] std::size_t top() const noexcept {
    return vertical_.lower_;
  }

  [[nodiscard]] std::size_t bottom() const noexcept {
    return vertical_.upper_;
  }

  [[nodiscard]] std::size_t height() const noexcept {
    return vertical_.size();
  }

private:
  details::limits horizontal_;
  details::limits vertical_;
};

namespace details {
  template<typename Edges>
  [[nodiscard]] box get_enclosure(Edges const& edges,
                                  std::size_t width) noexcept {
    limits horizontal;
    for (auto edge : edges) {
      horizontal.update(edge.position() % width);
    }

    return {
        horizontal,
        {edges.front().position() / width, edges.back().position() / width}};
  }

  template<cpl::pixel Ty>
  inline void write_pixels(Ty* output,
                           std::uint32_t left,
                           std::uint32_t right,
                           Ty color) noexcept {
    std::fill_n(
        output + left, static_cast<std::size_t>(right - left) + 1, color);
  }
} // namespace details

template<cpl::pixel Ty, typename Alloc = std::allocator<edge>>
class contour {
public:
  using pixel_type = Ty;
  using allocator_type = Alloc;
  using edges_t = std::vector<edge, allocator_type>;

public:
  inline contour(pixel_type const* base,
                 std::size_t width,
                 std::uint32_t id,
                 allocator_type const& allocator) noexcept
      : edges_{allocator}
      , base_{base}
      , width_{width}
      , id_{id} {
  }

  inline void add_point(pixel_type const* point, edge_side side) {
    ++area_;

    if (is_horisontal(side)) {
      edges_.push_back({point - base_, side});
      ++perimeter_;
    }
    else if (is_vertical(side)) {
      ++perimeter_;
    }
  }

  template<cpl::pixel Pixel>
  void recover(Pixel* output, Pixel color) const noexcept {
    sort();

    std::optional<std::uint32_t> left;
    for (auto edge : edges_) {
      if (is_right(edge.side())) {
        if (left) {
          details::write_pixels(output, left.value(), edge.position(), color);
          left.reset();
        }
        else {
          output[edge.position()] = color;
        }
      }
      else {
        left = edge.position();
      }
    }
  }

  inline void recover(pixel_type* output,
                      std::true_type /*unused*/) const noexcept {
    recover(output, color());
  }

  inline void recover(pixel_type* output,
                      std::false_type /*unused*/) const noexcept {
    auto c{color()};
    for (auto edge : edges_) {
      output[edge.position()] = c;
    }
  }

  [[nodiscard]] inline std::uint32_t area() const noexcept {
    return area_;
  }

  [[nodiscard]] inline std::uint32_t perimeter() const noexcept {
    return perimeter_;
  }

  [[nodiscard]] inline std::uint32_t id() const noexcept {
    return id_;
  }

  [[nodiscard]] inline box const& enclosure() const noexcept {
    if (!enclosure_) {
      sort();
      enclosure_ = details::get_enclosure(edges_, width_);
    }

    return *enclosure_;
  }

  [[nodiscard]] inline pixel_type color() const noexcept {
    if (!color_) {
      color_ = base_[edges_.front().position()];
    }

    return *color_;
  }

private:
  inline void sort() const noexcept {
    if (!sorted_) {
      std::sort(edges_.begin(), edges_.end());
      sorted_ = true;
    }
  }

private:
  mutable bool sorted_{};
  mutable edges_t edges_;

  pixel_type const* base_;
  std::size_t width_;

  std::uint32_t area_{0};
  std::uint32_t perimeter_{0};
  std::uint32_t id_;

  mutable std::optional<box> enclosure_;
  mutable std::optional<pixel_type> color_;
};

} // namespace ctr
