
#pragma once

namespace ifd {

template<typename Image>
struct frame {
  std::size_t number_;
  Image image_;
};

template<typename Ty>
concept feeder = requires(Ty a) {
  typename Ty::image_type;
  typename Ty::allocator_type;

  { a.has_more() }
  noexcept->std::same_as<bool>;

  { a.produce(std::declval<typename Ty::allocator_type>()) }
  ->std::same_as<frame<typename Ty::image_type>>;

  { a.produce() }
  ->std::same_as<frame<typename Ty::image_type>>;
};
} // namespace ifd
