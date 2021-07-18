
// standard image definitions

#pragma once

#include "cpl.hpp"
#include "mrl.hpp"

namespace sid {
namespace nat {
  using img_t = mrl::matrix<cpl::nat_cc>;

  template<typename Alloc>
  using imgal_t = mrl::matrix<cpl::nat_cc, Alloc>;
} // namespace nat
} // namespace sid
