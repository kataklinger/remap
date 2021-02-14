
#pragma once

#include "cdt.hpp"
#include "cte_v1.hpp"

#include <algorithm>
#include <unordered_map>

namespace mod {

template<ctr::pixel Ty, typename Alloc = std::allocator<char>>
class detector {
public:
  using pixel_type = Ty;
  using allocator_type = Alloc;

private:
  using source_t = cte::v1::outline_t<pixel_type>;
  using cell_t = typename source_t::value_type;

public:
  using contour_id = typename cell_t::id_type;
  using motion_t = std::tuple<contour_id, std::int32_t>;

private:
  using contour_type =
      ctr::contour<pixel_type, all::rebind_alloc_t<allocator_type, ctr::edge>>;
  using contours_type =
      std::vector<contour_type,
                  all::rebind_alloc_t<allocator_type, contour_type>>;

  using markings =
      std::vector<std::uint8_t,
                  all::rebind_alloc_t<allocator_type, std::uint8_t>>;

  struct motion_hash {
    [[nodiscard]] std::size_t
        operator()(motion_t const& motion) const noexcept {
      std::size_t hashed = 2166136261U;

      hashed ^= static_cast<std::size_t>(std::get<0>(motion));
      hashed *= 16777619U;

      hashed ^= static_cast<std::size_t>(std::get<1>(motion));
      hashed *= 16777619U;

      return hashed;
    }
  };

  using motion_counter_t =
      std::unordered_map<motion_t, std::uint16_t, motion_hash>;

  using motion_tracker_t =
      std::unordered_map<contour_id, motion_counter_t, std::hash<contour_id>>;

  using motion_map = std::unordered_map<contour_id, std::int32_t>;

public:
  detector(std::uint8_t margin,
           std::uint8_t window,
           allocator_type const& alloc = allocator_type{})
      : margin_{margin}
      , window_{window}
      , half_{static_cast<std::uint8_t>(window >> 1)}
      , tracker_{alloc} {
  }

  motion_map detect(source_t const& previous,
                    source_t const& current,
                    cdt::offset_t adjustment,
                    contours_type const& contours) {
    auto width{current.width()};

    auto [x, y]{adjustment};

    auto left{clip(x)}, right{clip(-x)}, top{clip(y)}, bottom{clip(-y)};

    auto p_start{previous.data() + left + top * width};
    auto c_start{current.data() + right + bottom * width};
    auto c_end{current.end() - top * width};
    auto c_size{width - left - right};

    auto marked{mark_motion(p_start, c_start, c_end, width, c_size, contours)};

    if (auto adj{std::max(0, -y)}; adj < half_) {
      auto v_off{adj * width};
      for (auto height{half_ + adj}; height < window_;
           c_start += width, p_start += width, ++height) {
        process_row(c_start,
                    c_start + c_size,
                    p_start,
                    v_off,
                    height,
                    x,
                    width,
                    marked);
      }
    }

    auto v_off{half_ * width};

    for (auto last{c_end - std::max(0, y)}; c_start < last;
         c_start += width, p_start += width) {
      process_row(
          c_start, c_start + c_size, p_start, v_off, window_, x, width, marked);
    }

    for (; c_start < c_end; c_start += width, p_start += width) {
      process_row(c_start,
                  c_start + c_size,
                  p_start,
                  v_off,
                  half_ + (c_end - c_start),
                  x,
                  width,
                  marked);
    }

    return refine(contours);
  }

private:
  [[nodiscard]] markings mark_motion(cell_t const* previous,
                                     cell_t const* current,
                                     cell_t const* end,
                                     std::size_t width,
                                     std::size_t window,
                                     contours_type const& contours) const {
    markings marked{contours.size(), tracker_.get_allocator()};

    for (; current < end; current += width, previous += width) {
      for (auto c{current}, e{current + window}, p{previous}; c < e; ++c, ++p) {
        if (c->color_ != p->color_ || c->edge_ != p->edge_) {
          marked[c->id_ - 1] = 1;
        }
      }
    }

    return marked;
  }

  void process_row(cell_t const* start,
                   cell_t const* end,
                   cell_t const* prev,
                   std::int32_t ver,
                   std::uint8_t height,
                   std::int32_t hor,
                   std::size_t width,
                   markings const& marked) {
    if (auto adj{std::max(0, -hor)}; adj < half_) {
      for (auto last{start + half_ - adj}, first{start}; start < last;
           ++start, ++prev) {

        if (marked[start->id_ - 1] != 0 &&
            start->edge_ != ctr::edge_side::none) {
          auto off_v{adj + (start - first)};
          auto width_w{half_ + off_v};

          process_window(prev - ver - off_v,
                         prev,
                         *start,
                         width_w,
                         height,
                         width - width_w);
        }
      }
    }

    auto adj{std::max(0, hor)};
    auto off_p{ver + half_};
    auto stride{width - window_};

    for (auto last{end + std::min(0, -half_ + adj)}; start < last;
         ++start, ++prev) {
      if (marked[start->id_ - 1] != 0 && start->edge_ != ctr::edge_side::none) {
        process_window(prev - off_p, prev, *start, window_, height, stride);
      }
    }

    for (; start < end; ++start, ++prev) {
      if (marked[start->id_ - 1] != 0 && start->edge_ != ctr::edge_side::none) {
        auto width_w{(end - start) + half_ + hor};
        process_window(
            prev - off_p, prev, *start, width_w, height, width - width_w);
      }
    }
  }

  void process_window(cell_t const* start,
                      cell_t const* center,
                      cell_t const& ref,
                      std::uint8_t width,
                      std::uint8_t height,
                      std::size_t stride) {
    auto& counter{tracker_[ref.id_]};
    for (; height > 0; --height, start += stride) {
      for (auto end{start + width}; start < end; ++start) {
        if (ref.edge_ == start->edge_ && ref.color_ == start->color_) {
          ++counter[{start->id_, static_cast<std::int16_t>(start - center)}];
        }
      }
    }
  }

  [[nodiscard]] inline std::size_t clip(std::int32_t edge) const noexcept {
    return static_cast<std::size_t>(std::max(edge, 0)) + margin_;
  }

  motion_map refine(contours_type const& contours) const {
    motion_map motions{tracker_.get_allocator()};

    for (auto& [id, offsets] : tracker_) {
      if (offsets.empty()) {
        continue;
      }

      auto [candidate, count] = *std::max_element(
          offsets.begin(), offsets.end(), [](auto& lhs, auto& rhs) {
            return lhs.second < rhs.second;
          });

      if (auto motion{std::get<1>(candidate)};
          motion != 0 && count > contours[id - 1].perimeter() / 2) {
        motions[id] = motion;
      }
    }

    return motions;
  }

private:
  std::uint8_t margin_;
  std::uint8_t window_;
  std::uint8_t half_;

  motion_tracker_t tracker_;
};
} // namespace mod
