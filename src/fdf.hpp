
// foreground filtering

#pragma once

#include "cfg.hpp"
#include "fgm.hpp"

namespace fdf {

namespace details {
  class frame {
  public:
    using grid_local_t = cfg::grid_local<std::allocator<char>>;

  public:
    [[nodiscard]] grid_local_t const& get_local() const noexcept {
      return keypoints_;
    }

    [[nodiscard]] cfg::grid_gloabal get_global() const {
    }

  private:
    mrl::matrix<cpl::nat_cc> image_;
    mrl::matrix<std::uint8_t> mask_;
    grid_local_t keypoints_;
  };

  class segment {
  public:
    using fragment_t = fgm::fragment<cfg::color_depth>;

  public:
    segment(fragment_t const& fragment)
        : original_{fragment.blend()} {
    }

    void match(frame const& update) {
    }

    void accept(frame& update) {
    }

  private:
    fgm::point_t position_{};

    mrl::matrix<nat_cc> current_image_;
    mrl::matrix<std::uint8_t> current_mask_;
    grid_local_t current_keypoints_;

    mrl::matrix<cpl::nat_cc> original_image_;
    mrl::matrix<std::uint8_t> original_mask_;
    cfg::grid_gobal original_keypoints_;
  };
} // namespace details
} // namespace fdf
