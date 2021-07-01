
// native-color c64 compression

#include "cpl.hpp"
#include "mrl.hpp"

namespace ncc {

[[nodiscard]] std::vector<std::uint8_t>
    compress(mrl::matrix<cpl::nat_cc> const& image);

[[nodiscard]] mrl::matrix<cpl::nat_cc>
    decompress(std::vector<std::uint8_t> const& pack,
               mrl::dimensions_t const& dim);

} // namespace ncc
