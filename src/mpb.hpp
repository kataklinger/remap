
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

public:
  inline builder(adapter_type const& adapter) noexcept
      : adapter_{adapter} {
  }

  [[nodiscard]] std::vector<sid::nat::dimg_t> build() {
    auto& callbacks{adapter_.get_callbacks()};

    auto window{aws::scan(
        adapter_.get_feed(), adapter_.get_screen_dimensions(), callbacks)};

    callbacks(window);

    if (!window) {
      return {};
    }

    auto window_dim{window->bounds().dimensions()};

    frc::collector collector{window_dim};
    collector.collect(adapter_.get_feed(window->margins()),
                      adapter_.get_compression(),
                      callbacks);

    auto fragments{collector.complete()};

    callbacks(fragments);

    auto spliced{
        fgs::splice<frc::color_depth>(fragments.begin(), fragments.end())};

    callbacks(spliced);

    auto filtered{fdf::filter(
        spliced, window_dim, adapter_.get_compression(), callbacks)};

    callbacks(filtered);

    std::vector<sid::nat::dimg_t> cleaned{filtered.size()};
    std::transform(
        std::execution::par,
        filtered.begin(),
        filtered.end(),
        cleaned.begin(),
        [&callbacks, dev = adapter_.get_artifact_filter_dev()](auto& fragment) {
          return arf::filter(fragment,
                             callbacks,
                             dev,
                             typename adapter_type::artifact_filter_size{});
        });

    return cleaned;
  }

private:
  adapter_type adapter_;
};
} // namespace mpb
