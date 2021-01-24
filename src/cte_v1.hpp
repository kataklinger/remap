
#pragma once

#include "all.hpp"
#include "cpl.hpp"
#include "mrl.hpp"

#include <deque>
#include <optional>
#include <queue>
#include <vector>

namespace cte {
namespace v1 {
  inline constexpr std::uint32_t horizon_id{0xffffff};

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

    inline void set_pixels(cpl::nat_cc* output,
                           std::uint32_t left,
                           std::uint32_t right,
                           cpl::nat_cc color) noexcept {
      std::memset(output + left, color.value, right - left + 1);
    }
  } // namespace details

  template<typename Alloc = std::allocator<edge>>
  class contour {
  public:
    using allocator_type = Alloc;
    using edges_t = std::vector<edge, allocator_type>;

  public:
    inline contour(cpl::nat_cc const* base,
                   std::size_t width,
                   std::uint32_t id,
                   allocator_type const& allocator) noexcept
        : edges_{allocator}
        , base_{base}
        , width_{width}
        , id_{id} {
    }

    inline void
        add_point(cpl::nat_cc const* point, bool left_edge, bool right_edge) {
      ++area_;

      if (left_edge || right_edge) {
        edges_.push_back({point - base_, right_edge});
      }
    }

    void recover(cpl::nat_cc* output,
                 std::true_type /*unused*/) const noexcept {
      sort();
      auto c{color()};

      std::optional<std::uint32_t> left;
      for (auto edge : edges_) {
        if (edge.is_right()) {
          if (left) {
            details::set_pixels(output, left.value(), edge.position(), c);
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

    void recover(cpl::nat_cc* output,
                 std::false_type /*unused*/) const noexcept {
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

    [[nodiscard]] inline cpl::nat_cc color() const noexcept {
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

    cpl::nat_cc const* base_;
    std::size_t width_;

    std::uint32_t area_{0};
    std::uint32_t id_;

    mutable std::optional<box> enclosure_;
    mutable std::optional<cpl::nat_cc> color_;
  };

  template<typename Alloc = std::allocator<edge>>
  class extractor {
  public:
    using allocator_type = Alloc;
    using contour_type = contour<allocator_type>;

    using contours =
        std::vector<contour_type,
                    all::rebind_alloc_t<allocator_type, contour_type>>;

  private:
    struct state {
      std::uint32_t id_ : 24;
      std::uint32_t edge_ : 1;
    };

    using path_node = cpl::nat_cc const*;
    using path_containter =
        std::deque<path_node, all::rebind_alloc_t<allocator_type, path_node>>;
    using path_type = std::queue<path_node, path_containter>;

  public:
    inline extractor(mrl::size_type width,
                     mrl::size_type height,
                     allocator_type const& allocator)
        : allocator_{allocator}
        , walk_{width, height}
        , path_{path_containter{allocator}} {
    }

  public:
    [[nodiscard]] contours extract(mrl::matrix<cpl::nat_cc> const& image) {
      clear_walk();

      contours extracted{allocator_};
      for (auto position{image.data() + image.width() + 1},
           last{image.end() - image.width() + 1};
           position < last;
           position += image.width()) {

        process_row(image.data(), position, extracted);
      }

      return extracted;
    }

  private:
    inline void process_row(cpl::nat_cc const* image,
                            cpl::nat_cc const* position,
                            contours& output) {
      std::uint32_t id{0};
      auto walk{walk_.data()};

      for (auto last{position + walk_.width() - 2}; position < last;
           ++position) {
        if (walk[position - image].id_ == 0) {
          output.push_back(extract_single(image, position, ++id));
        }
      }
    }

    [[nodiscard]] contour_type extract_single(cpl::nat_cc const* image,
                                              cpl::nat_cc const* position,
                                              std::uint32_t id) {
      auto width{walk_.width()};
      auto walk{walk_.data()};

      contour_type result{image, width, id, allocator_};

      path_.push(position);
      while (!path_.empty()) {
        auto pixel{path_.front()};
        auto cell{walk + (pixel - image)};

        push_pixel(pixel, cell, id, -width);
        push_pixel(pixel, cell, id, +width);

        auto left{push_pixel(pixel, cell, id, -1)};
        auto right{push_pixel(pixel, cell, id, +1)};

        result.add_point(pixel, left, right);

        path_.pop();
      }

      return result;
    }

    bool push_pixel(cpl::nat_cc const* pixel,
                    state* cell,
                    std::uint32_t id,
                    std::ptrdiff_t offset) {
      if (auto n_pixel{pixel + offset}; *n_pixel == *pixel) {
        auto n_cell{cell + offset};
        if (n_cell->id_ == 0) {
          n_cell->id_ = id;
          path_.push(n_pixel);
        }

        return n_cell->id_ == horizon_id;
      }

      return true;
    }

    void clear_walk() noexcept {
      auto first{walk_.data()};
      for (auto last{walk_.data() + walk_.width()}; first < last; ++first) {
        *first = {.id_ = horizon_id};
      }

      auto size{(walk_.width() - 2) * walk_.height()};
      std::memset(first, 0, size * sizeof(state));

      for (auto last{first + size - walk_.width()}; first <= last;
           first += walk_.width()) {
        *first = *(first + walk_.width() - 1) = {.id_ = horizon_id};
      }

      for (auto last{walk_.end()}; first < last; ++first) {
        *first = {.id_ = horizon_id};
      }
    }

  private:
    [[no_unique_address]] allocator_type allocator_;

    mrl::matrix<state> walk_;
    path_type path_;
  };
} // namespace v1
} // namespace cte
