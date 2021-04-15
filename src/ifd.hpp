
#pragma once

namespace ifd {
template<typename Ty>
concept feeder = requires(Ty a) {
  typename Ty::image_type;
  typename Ty::allocator_type;

  { a.has_more() }
  noexcept->std::same_as<bool>;

  { a.produce(std::declval<typename Ty::allocator_type>()) }
  ->std::same_as<typename Ty::image_type>;

  { a.produce() }
  ->std::same_as<typename Ty::image_type>;
};
} // namespace ifd
