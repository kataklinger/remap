
#pragma once

#include "cpl.hpp"

#include <optional>
#include <vector>

namespace ctr {
template<typename Ty>
concept pixel = requires(Ty v) {
  requires std::is_trivial_v<Ty>;
  requires std::totally_ordered<Ty>;
  requires cpl::pixel_color<decltype(value(v))>;
};

class edge {
public:
  inline edge(std::ptrdiff_t position, bool is_right) noexcept
      : rep_{static_cast<std::uint32_t>(position << 1) |
             static_cast<std::uint32_t>(is_right)} {
  }

  [[nodiscard]] inline std::uint32_t position() const noexcept {
    return rep_ >> 1;
  }

  [[nodiscard]] inline bool is_right() const noexcept {
    return (rep_ & 1) != 0;
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

  template<pixel Ty>
  inline void write_pixels(Ty* output,
                           std::uint32_t left,
                           std::uint32_t right,
                           Ty color) noexcept {
    std::memset(output + left,
                value(color),
                static_cast<std::size_t>(right - left) + 1);
  }
} // namespace details

template<pixel Ty, typename Alloc = std::allocator<edge>>
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

  inline void
      add_point(pixel_type const* point, bool left_edge, bool right_edge) {
    ++area_;

    if (left_edge || right_edge) {
      edges_.push_back({point - base_, right_edge});
    }
  }

  void recover(pixel_type* output, std::true_type /*unused*/) const noexcept {
    sort();
    auto c{color()};

    std::optional<std::uint32_t> left;
    for (auto edge : edges_) {
      if (edge.is_right()) {
        if (left) {
          details::write_pixels(output, left.value(), edge.position(), c);
          left.reset();
        }
        else {
          output[edge.position()] = c;
        }
      }
      else {
        left = edge.position();
      }
    }
  }

  void recover(pixel_type* output, std::false_type /*unused*/) const noexcept {
    auto c{color()};
    for (auto edge : edges_) {
      output[edge.position()] = c;
    }
  }

  [[nodiscard]] inline std::uint32_t area() const noexcept {
    return area_;
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
  std::uint32_t id_;

  mutable std::optional<box> enclosure_;
  mutable std::optional<pixel_type> color_;
};

} // namespace ctr
