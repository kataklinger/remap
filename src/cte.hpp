
#pragma once

#include "all.hpp"
#include "ctr.hpp"
#include "mrl.hpp"

#include <deque>
#include <queue>

namespace cte {
inline constexpr std::uint16_t horizon_id{0xffff};

template<cpl::pixel Ty>
struct cell {
  using pixel_type = Ty;
  using id_type = std::uint16_t;

  id_type id_;
  pixel_type color_;
  ctr::edge_side edge_;
};

template<cpl::pixel Ty>
using outline_t = mrl::matrix<cell<Ty>>;

template<cpl::pixel Ty, typename Alloc = std::allocator<ctr::edge>>
class extractor {
public:
  using pixel_type = Ty;
  using allocator_type = Alloc;
  using contour_type = ctr::contour<pixel_type, allocator_type>;

  using contours =
      std::vector<contour_type,
                  all::rebind_alloc_t<allocator_type, contour_type>>;

  using cell_type = cell<pixel_type>;
  static_assert(std::is_trivially_copyable_v<cell_type>);

  using outline_type = outline_t<pixel_type>;

private:
  using path_node = pixel_type const*;
  using path_containter =
      std::deque<path_node, all::rebind_alloc_t<allocator_type, path_node>>;
  using path_type = std::queue<path_node, path_containter>;

public:
  inline extractor(mrl::size_type width,
                   mrl::size_type height,
                   allocator_type const& allocator)
      : allocator_{allocator}
      , outline_{width, height}
      , path_{path_containter{allocator}} {
  }

public:
  [[nodiscard]] contours extract(mrl::matrix<pixel_type> const& image) {
    clear_outline();

    contours extracted{allocator_};
    for (auto position{image.data() + image.width() + 1},
         last{image.end() - image.width() + 1};
         position < last;
         position += image.width()) {

      process_row(image.data(), position, extracted);
    }

    return extracted;
  }

  [[nodiscard]] outline_type const& outline() const noexcept {
    return outline_;
  }

private:
  inline void process_row(pixel_type const* image,
                          pixel_type const* position,
                          contours& output) {
    auto outline{outline_.data()};

    for (auto last{position + outline_.width() - 2}; position < last;
         ++position) {
      if (outline[position - image].id_ == 0) {
        output.push_back(extract_single(
            image,
            position,
            static_cast<typename cell_type::id_type>(output.size() + 1)));
      }
    }
  }

  [[nodiscard]] contour_type extract_single(pixel_type const* image,
                                            pixel_type const* position,
                                            std::uint32_t id) {
    auto width{outline_.width()};
    auto outline{outline_.data()};

    contour_type result{image, width, id, allocator_};

    path_.push(position);
    outline[(position - image)].id_ = id;

    while (!path_.empty()) {
      auto pixel{path_.front()};
      auto cell{outline + (pixel - image)};

      cell->color_ = *pixel;
      cell->edge_ = ctr::create_edge(push_pixel(pixel, cell, id, -1),
                                     push_pixel(pixel, cell, id, +1),
                                     push_pixel(pixel, cell, id, -width),
                                     push_pixel(pixel, cell, id, +width));

      result.add_point(pixel, cell->edge_);

      path_.pop();
    }

    return result;
  }

  bool push_pixel(pixel_type const* pixel,
                  cell_type* cell,
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

  void clear_outline() noexcept {
    auto first{outline_.data()};
    for (auto last{outline_.data() + outline_.width()}; first < last; ++first) {
      *first = {.id_ = horizon_id};
    }

    auto size{(outline_.width() - 2) * outline_.height()};
    std::memset(first, 0, size * sizeof(cell_type));

    for (auto last{first + size - 2 * outline_.width()}; first <= last;
         first += outline_.width()) {
      *first = *(first + outline_.width() - 1) = {.id_ = horizon_id};
    }

    for (auto last{outline_.end()}; first < last; ++first) {
      *first = {.id_ = horizon_id};
    }
  }

private:
  [[no_unique_address]] allocator_type allocator_;

  outline_type outline_;
  path_type path_;
};
} // namespace cte
