
// map building

#include "aws.hpp"
#include "fdf.hpp"

namespace mpb {

class config {
public:
  using feeder_type = Feeder;

  static constexpr mrl::dimension_t screen_size{388, 312};
  static constexpr std::uint8_t artifact_filter_size{15};
  static constexpr float artifact_filter_dev{2.0f};

public:
  [[nodiscard]] feeder_type begin_feed() const {
  }

  [[nodiscard]] feeder_type begin_feed(mrl::region_t crop) const {
  }

  [[nodiscard]] std::vector<std::uint8_t>
      compress(mrl::matrix<cpl::nat_cc> const& image) const {
  }

  [[nodiscard]] mrl::matrix<cpl::nat_cc>
      decompress(std::vector<std::uint8_t> const& compressed,
                 mrl::dimensions_t const& dim) const {
  }

  [[nodiscard]] mrl::dimension_t get_screen_size() const noexcept {
    return screen_size;
  }

  [[nodiscard]] std::uint8_t get_artifact_filter_size() const noexcept {
    return artifact_filter_size;
  }

  [[nodiscard]] float get_artifact_filter_dev() const noexcept {
    return artifact_filter_dev;
  }

public:
};

template<typename Config>
class builder {
public:
  using config_type = Config;

public:
  inline builder(config_type const& config) noexcept
      : config_{config} {
  }

  [[nodiscard]] std::vector<mrl::matrix<cpl::nat_cc>> build() {
    auto window{aws::scan(config_.begin_feed(), config_.get_screen_size()))};
    if (!window) {
      return {};
    }

    auto window_dim{window->bounds().dimensions()};

    frc::collector collector{window_dim};
    collector.collect(config.begin_feed(window->margins()),
                      [&config_](auto& img) { return config_.compress(img); });

    auto fragments{collector.complete()};

    auto spliced{fgs::splice<frc::collector::color_depth>(fragments.begin(),
                                                          fragments.end())};
    auto filtered{fdf::filter(
        spliced, window_dim, [&config_](auto const& img, auto const& dim) {
          return config_.decompress(img, dim);
        })};

    std::vector<mrl::matrix<cpl::nat_cc>> cleaned{filtered.size()};
    std::transform(std::execution::par,
                   filtered.begin(),
                   filtered.end(),
                   cleaned.begin(),
                   [&config_](auto& fragment) {
                     return arf::filter(
                         fragment,
                         config_.get_artifact_filter_dev(),
                         arf::filter_size<config_type::artifact_filter_size>{});
                   });

    return cleared;
  }

private:
  config_type config_;
};
} // namespace mpb
