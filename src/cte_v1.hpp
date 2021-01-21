
#pragma once

#include "all.hpp"
#include "cpl.hpp"
#include "mrl.hpp"

#include <deque>
#include <queue>
#include <vector>

namespace cte {
namespace v1 {
  namespace details {} // namespace details

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

  private:
    std::uint32_t rep_;
  };

  template<typename Alloc = std::allocator<edge>>
  class contour {
  public:
    using allocator_type = Alloc;

  public:
    inline contour(cpl::nat_cc const* base,
                   std::uint32_t id,
                   allocator_type const& allocator) noexcept
        : edges_{allocator}
        , base_{base}
        , id_{id} {
    }

    inline void
        add_point(cpl::nat_cc const* point, bool left_edge, bool right_edge) {
      ++area_;

      if (left_edge || right_edge) {
        edges_.push_back({point - base_, right_edge});
      }
    }

    [[nodiscard]] inline std::uint32_t area() const noexcept {
      return area_;
    }

    [[nodiscard]] inline std::uint32_t id() const noexcept {
      return id_;
    }

  private:
    std::vector<edge, allocator_type> edges_;
    cpl::nat_cc const* base_;

    std::uint32_t area_{0};
    std::uint32_t id_;
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
        , state_{width, height}
        , path_{path_containter{allocator}} {
    }

  public:
    [[nodiscard]] contours extract(mrl::matrix<cpl::nat_cc> const& image) {
      clear_state();

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

      for (auto last{position + state_.width() - 2}; position < last;
           ++position) {
        if (state_.data()[position - image].id_ == 0) {
          output.push_back(extract_single(image, position, ++id));
        }
      }
    }

    [[nodiscard]] contour_type extract_single(cpl::nat_cc const* image,
                                              cpl::nat_cc const* position,
                                              std::uint32_t id) {
      contour_type result{image, id, allocator_};

      path_.push(position);
      while (!path_.empty()) {
        auto pixel{path_.front()};
        auto cell{state_.data() + (pixel - image)};

        push_pixel(pixel, cell, id, -state_.width());
        push_pixel(pixel, cell, id, +state_.width());

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

    void clear_state() noexcept {
      auto first{state_.data()};
      for (auto last{state_.data() + state_.width()}; first < last; ++first) {
        *first = {.id_ = horizon_id};
      }

      auto size{(state_.width() - 2) * state_.height()};
      std::memset(first, 0, size * sizeof(state));

      for (auto last{first + size - state_.width()}; first <= last;
           first += state_.width()) {
        *first = *(first + state_.width() - 1) = {.id_ = horizon_id};
      }

      for (auto last{state_.end()}; first < last; ++first) {
        *first = {.id_ = horizon_id};
      }
    }

  private:
    [[no_unique_address]] allocator_type allocator_;

    mrl::matrix<state> state_;
    path_type path_;
  };
} // namespace v1
} // namespace cte
