
// frame collection

#pragma once

#include "fgm.hpp"
#include "ifd.hpp"
#include "kpe.hpp"
#include "kpm.hpp"

#include <execution>
#include <list>

namespace frc {

template<typename Ty>
using allocator_t = all::frame_allocator<Ty>;

using image_type = sid::nat::aimg_t<allocator_t<cpl::nat_cc>>;
using frame_type = ifd::frame<image_type>;

static inline constexpr std::size_t grid_horizontal{4};
static inline constexpr std::size_t grid_vertical{2};
static inline constexpr std::size_t grid_overlap{16};

using grid_type = kpr::grid<grid_horizontal, grid_vertical, allocator_t<char>>;

class collector {
private:
  struct match_config {
    using allocator_type = allocator_t<char>;
    static constexpr std::size_t weight_switch{10};
    static constexpr std::size_t region_votes{3};

    inline explicit match_config(allocator_type const& alloc) noexcept
        : allocator_{alloc} {
    }

    [[nodiscard]] inline allocator_type get_allocator() const noexcept {
      return allocator_;
    }

    [[no_unique_address]] allocator_type allocator_;
  };

  using keypoint_extractor_t = kpe::extractor<grid_type, grid_overlap>;

  using pixel_alloc_t = allocator_t<cpl::nat_cc>;

public:
  collector(mrl::dimensions_t dimensions)
      : extractor_{dimensions} {
  }

  template<typename Feeder, typename Comp, typename Callback>
  void collect(Feeder&& feed, Comp&& comp, Callback&& cb) requires(
      ifd::feeder<std::decay_t<Feeder>, pixel_alloc_t>&&
          icd::compressor<std::decay_t<Comp>, pixel_alloc_t>) {
    if (feed.has_more()) {
      all::memory_stack<cpl::nat_cc> memory{};

      auto pkeys{process_init(feed, comp, memory.previous())};
      for (std::int32_t x{0}, y{0}; feed.has_more();) {
        all::memory_swing swing{memory};
        pkeys = process_frame(feed, comp, cb, pkeys, swing);
      }
    }
  }

  [[nodiscard]] inline fgm::fragment const& current() const noexcept {
    return *current_;
  }

  [[nodiscard]] inline std::list<fgm::fragment> complete() noexcept {
    for (auto& fragment : fragments_) {
      fragment.normalize();
    }

    return std::move(fragments_);
  }

private:
  template<typename Feed, typename Comp>
  auto process_init(Feed& feed, Comp& comp, pixel_alloc_t const& alloc) {
    auto frame{feed.produce(alloc)};

    add_fragment(frame.image_.dimensions());

    image_type median{frame.image_.dimensions(), alloc};
    auto result{extractor_.extract(frame.image_, median, alloc)};

    blit(comp, frame, median);

    return result;
  }

  template<typename Feed, typename Comp, typename Callback>
  auto process_frame(Feed& feed,
                     Comp& comp,
                     Callback&& cb,
                     grid_type const& previous,
                     pixel_alloc_t const& alloc) {
    auto frame{feed.produce(alloc)};
    auto& dim{frame.image_.dimensions()};

    image_type median{dim, alloc};
    auto keys{extractor_.extract(frame.image_, median, alloc)};

    if (auto off{kpm::match(match_config{alloc}, previous, keys)}; off) {
      position_.x_ += off->x_;
      position_.y_ += off->y_;
    }
    else {
      add_fragment(dim);
    }

    blit(comp, frame, median);

    cb(*current_, frame, median, keys);

    return keys;
  }

  inline void add_fragment(mrl::dimensions_t dimension) {
    current_ = &fragments_.emplace_back(dimension);
    position_.x_ = position_.y_ = 0;
  }

  template<typename Comp>
  inline void blit(Comp& comp,
                   ifd::frame<image_type> const& frame,
                   image_type const& median) noexcept {
    auto& [no, image]{frame};
    current_->blit(position_, image, {comp(image), comp(median)}, no);
  }

private:
  keypoint_extractor_t extractor_;

  fgm::point_t position_{};

  std::list<fgm::fragment> fragments_;
  fgm::fragment* current_{nullptr};
};

} // namespace frc
