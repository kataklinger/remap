
// standard image definitions

#pragma once

#include "cpl.hpp"
#include "mrl.hpp"

namespace sid {
namespace nat {
  using dimg_t = mrl::matrix<cpl::nat_cc>;

  template<typename Alloc>
  using aimg_t = mrl::matrix<cpl::nat_cc, Alloc>;
} // namespace nat

namespace mon {
  using dimg_t = mrl::matrix<cpl::mon_bv>;

  template<typename Alloc>
  using aimg_t = mrl::matrix<cpl::mon_bv, Alloc>;
}
} // namespace sid
