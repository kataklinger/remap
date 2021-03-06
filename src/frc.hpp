
// frame collection

#pragma once

#include "all.hpp"
#include "fgm.hpp"
#include "ifd.hpp"
#include "kpe.hpp"
#include "kpm.hpp"
#include "kpr.hpp"

#include <execution>
#include <list>

namespace frc {
class collector {
public:
  static inline constexpr std::uint8_t color_depth{16};

  template<typename Ty>
  using allocator_t = all::frame_allocator<Ty>;

  using image_type = mrl::matrix<cpl::nat_cc, allocator_t<cpl::nat_cc>>;

  using fragment_t = fgm::fragment<color_depth>;
  using fragment_list = std::list<fragment_t>;

private:
  static inline constexpr std::size_t grid_horizontal{4};
  static inline constexpr std::size_t grid_vertical{2};
  static inline constexpr std::size_t grid_overlap{16};

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

  using grid_type =
      kpr::grid<grid_horizontal, grid_vertical, allocator_t<char>>;
  using keypoint_extractor_t = kpe::extractor<grid_type, grid_overlap>;

  using pixel_alloc_t = allocator_t<cpl::nat_cc>;

public:
  collector(mrl::dimensions_t dimensions)
      : extractor_{dimensions} {
  }

  template<typename Feeder, typename Comp>
  void collect(Feeder&& feed,
               Comp&& comp) requires(ifd::feeder<std::decay_t<Feeder>>) {
    if (feed.has_more()) {
      all::memory_pool ppool{0};

      auto pkeys{process_init(feed, comp, pixel_alloc_t{ppool})};
      for (std::int32_t x{0}, y{0}; feed.has_more();) {
        all::memory_pool cpool{ppool.total_used() << 1};
        pkeys = process_frame(feed, comp, pkeys, pixel_alloc_t{cpool});
        ppool = std::move(cpool);
      }
    }
  }

  [[nodiscard]] inline fragment_t const& current() const noexcept {
    return *current_;
  }

  [[nodiscard]] inline fragment_list complete() noexcept {
    return std::move(fragments_);
  }

private:
  template<typename Feed, typename Comp>
  auto process_init(Feed& feed, Comp& comp, pixel_alloc_t const& alloc) {
    auto [no, image]{feed.produce(alloc)};

    add_fragment(image.dimensions());

    image_type median{image.dimensions(), alloc};
    auto result{extractor_.extract(image, median, alloc)};

    blit(comp, image, median, no);

    return result;
  }

  template<typename Feed, typename Comp>
  auto process_frame(Feed& feed,
                     Comp& comp,
                     grid_type const& previous,
                     pixel_alloc_t const& alloc) {
    auto [no, image]{feed.produce(alloc)};

    image_type median{image.dimensions(), alloc};
    auto keys{extractor_.extract(image, median, alloc)};

    if (auto off{kpm::match(match_config{alloc}, previous, keys)}; off) {
      position_.x_ += off->x_;
      position_.y_ += off->y_;
    }
    else {
      add_fragment(image.dimensions());
    }

    blit(comp, image, median, no);

    return keys;
  }

  inline void add_fragment(mrl::dimensions_t dimension) {
    current_ = &fragments_.emplace_back(dimension);
    position_.x_ = position_.y_ = 0;
  }

  template<typename Comp>
  inline void blit(Comp& comp,
                   image_type const& image,
                   image_type const& median,
                   std::size_t frame_no) noexcept {
    current_->blit(position_, image, {comp(image), comp(median)}, frame_no);
  }

private:
  keypoint_extractor_t extractor_;

  fgm::point_t position_{};

  fragment_list fragments_;
  fragment_t* current_{nullptr};
};
} // namespace frc
