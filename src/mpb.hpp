
// map building

#pragma once

#include "arf.hpp"
#include "aws.hpp"
#include "fdf.hpp"
#include "fgs.hpp"
#include "frc.hpp"

namespace mpb {

template<typename Adapter>
class builder {
public:
  using adapter_type = Adapter;

private:
  using callbacks_type = typename Adapter::callbacks_type;
  using feed_type = typename Adapter::feed_type;

public:
  inline builder(adapter_type const& adapter) noexcept
      : adapter_{adapter} {
  }

  [[nodiscard]] std::vector<sid::nat::dimg_t> build() {
    if (auto window{get_window()}; window) {
      auto dimensions{window->bounds().dimensions()};

      auto feed{adapter_.get_feed(window->margins())};

      auto fragments{collect(feed, dimensions)};
      auto spliced{splice(fragments)};
      auto filtered{filter(dimensions, spliced)};
      return clean(filtered);
    }

    return {};
  }

private:
  [[nodiscard]] inline auto get_window() {
    auto result{
        aws::scan(adapter_.get_feed(), adapter_.get_screen_dimensions(), cb())};

    cb()(result);
    return result;
  }

  [[nodiscard]] inline auto collect(feed_type& feed,
                                    mrl::dimensions_t const& window) {
    frc::collector collector{window};

    collector.collect(feed, adapter_.get_compression(), cb());
    auto result{collector.complete()};

    cb()(result);
    return result;
  }

  [[nodiscard]] inline auto splice(std::list<fgm::fragment>& fragments) {
    auto result{fgs::splice(fragments.begin(), fragments.end())};

    cb()(result);
    return result;
  }

  [[nodiscard]] inline auto filter(mrl::dimensions_t const& window,
                                   std::list<fgm::fragment>& fragments) {
    auto result{
        fdf::filter(fragments, window, adapter_.get_compression(), cb())};

    cb()(result);
    return result;
  }

  [[nodiscard]] inline auto clean(std::vector<fgm::fragment>& fragments) {
    std::vector<sid::nat::dimg_t> result{fragments.size()};
    std::transform(
        std::execution::par,
        fragments.begin(),
        fragments.end(),
        result.begin(),
        [this, dev = adapter_.get_artifact_filter_dev()](auto& fragment) {
          return arf::filter(fragment,
                             cb(),
                             dev,
                             typename adapter_type::artifact_filter_size{});
        });

    return result;
  }

  [[nodiscard]] inline auto& cb() noexcept {
    return adapter_.get_callbacks();
  }

private:
  adapter_type adapter_;
};
} // namespace mpb
