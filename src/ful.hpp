
// fragment utility library

#include "fgm.hpp"

#include <filesystem>
#include <fstream>

namespace ful {

template<typename Iter>
void write(std::filesystem::path dir, Iter first, Iter last) {
  for (auto i{0}; first != last; ++first, ++i) {
    std::fstream output{dir / std::to_string(i),
                        std::ios::out | std::ios::binary};

    auto dim{first->dots().dimensions()};
    output.write(reinterpret_cast<char const*>(&dim), sizeof(dim));

    output.write(reinterpret_cast<char const*>(first->dots().data()),
                 first->dots().dimensions().area() * sizeof(fgm::dot_type));

    auto zero{first->zero()};
    output.write(reinterpret_cast<char const*>(&zero), sizeof(zero));

    auto frames{first->frames().size()};
    output.write(reinterpret_cast<char const*>(&frames), sizeof(frames));

    for (auto& frame : first->frames()) {
      output.write(reinterpret_cast<char const*>(&frame.number_),
                   sizeof(frame.number_));
      output.write(reinterpret_cast<char const*>(&frame.position_),
                   sizeof(frame.position_));

      auto size_i{frame.data_.image_.size()};
      output.write(reinterpret_cast<char const*>(&size_i), sizeof(size_i));
      output.write(reinterpret_cast<char const*>(frame.data_.image_.data()),
                   size_i);

      auto size_m{frame.data_.median_.size()};
      output.write(reinterpret_cast<char const*>(&size_m), sizeof(size_m));
      output.write(reinterpret_cast<char const*>(frame.data_.median_.data()),
                   size_m);
    }
  }
}

[[nodiscard]] auto read(std::filesystem::path dir) {
  using namespace std::filesystem;

  std::vector<std::filesystem::path> files;
  std::copy(
      directory_iterator(dir), directory_iterator(), back_inserter(files));
  std::sort(files.begin(), files.end(), [](auto& a, auto& b) {
    return std::stoi(a.filename().string()) < std::stoi(b.filename().string());
  });

  std::vector<fgm::fragment> result;
  for (auto& file : files) {
    std::ifstream input{file, std::ios::in | std::ios::binary};

    mrl::dimensions_t dim{};
    input.read(reinterpret_cast<char*>(&dim), sizeof(mrl::dimensions_t));

    fgm::fragment::matrix_type temp{dim};
    input.read(reinterpret_cast<char*>(temp.data()),
               dim.area() * sizeof(fgm::dot_type));

    fgm::point_t zero{};
    input.read(reinterpret_cast<char*>(&zero), sizeof(zero));

    std::size_t count{};
    input.read(reinterpret_cast<char*>(&count), sizeof(count));

    std::vector<fgm::frame> frames{};
    for (std::size_t i{0}; i < count; ++i) {
      fgm::frame frame{};

      input.read(reinterpret_cast<char*>(&frame.number_),
                 sizeof(frame.number_));
      input.read(reinterpret_cast<char*>(&frame.position_),
                 sizeof(frame.position_));

      std::size_t size_i{};
      input.read(reinterpret_cast<char*>(&size_i), sizeof(size_i));
      frame.data_.image_.resize(size_i);
      input.read(reinterpret_cast<char*>(frame.data_.image_.data()), size_i);

      std::size_t size_m{};
      input.read(reinterpret_cast<char*>(&size_m), sizeof(size_m));
      frame.data_.median_.resize(size_m);
      input.read(reinterpret_cast<char*>(frame.data_.median_.data()), size_m);

      frames.push_back(frame);
    }

    result.emplace_back(
        std::move(temp), mrl::dimensions_t{1, 1}, zero, std::move(frames));
  }

  return result;
}

} // namespace ful
