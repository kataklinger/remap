
#include <memory>

namespace all {
template<typename Alloc, typename Ty>
using rebind_alloc_t =
    typename std::allocator_traits<Alloc>::template rebind_alloc<Ty>;
}
