
// map building

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

  [[nodiscard]] std::vector<mrl::matrix<cpl::nat_cc>> build() {
    auto window{
        aws::scan(adapter_.get_feed(), adapter_.get_screen_dimensions())};
    if (!window) {
      return {};
    }

    auto window_dim{window->bounds().dimensions()};

    frc::collector collector{window_dim};
    collector.collect(adapter_.get_feed(window->margins()),
                      [this](auto& img) { return adapter_.compress(img); });

    auto fragments{collector.complete()};

    auto spliced{fgs::splice<frc::collector::color_depth>(fragments.begin(),
                                                          fragments.end())};
    auto filtered{fdf::filter(
        spliced, window_dim, [this](auto const& img, auto const& dim) {
          return adapter_.decompress(img, dim);
        })};

    std::vector<mrl::matrix<cpl::nat_cc>> cleaned{filtered.size()};
    std::transform(
        std::execution::par,
        filtered.begin(),
        filtered.end(),
        cleaned.begin(),
        [this](auto& fragment) {
          return arf::filter(
              fragment,
              adapter_.get_artifact_filter_dev(),
              arf::filter_size<adapter_type::artifact_filter_size>{});
        });

    return cleaned;
  }

private:
  adapter_type adapter_;
};
} // namespace mpb
