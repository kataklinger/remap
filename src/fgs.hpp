
// fragment splicing

#pragma once

#include "fgm.hpp"
#include "kpe.hpp"
#include "kpm.hpp"

#include <execution>
#include <stack>

namespace fgs {

namespace details {

  using grid_t = kpr::grid<1, 1, std::allocator<char>>;

  struct delta;
  struct snippet;

  using edges_t = std::list<delta>;
  using snippet_iterator_t = std::list<snippet>::iterator;

  struct snippet {
    [[nodiscard]] inline void bind(snippet_iterator_t self,
                                   snippet_iterator_t other,
                                   kpm::vote const& vote);

    inline void unbind();

    [[nodiscard]] inline edges_t::iterator
        add(bool primary, kpm::vote const& vote, snippet_iterator_t other);

    fgm::fragment fragment_;
    sid::mon::dimg_t mask_;

    grid_t grid_;

    edges_t edges_;
  };

  struct delta {
    delta(bool primary, kpm::vote const& vote, snippet_iterator_t other)
        : primary_{primary}
        , vote_{primary ? vote : vote.reverse()}
        , other_{other} {
    }

    bool primary_;

    kpm::vote vote_;

    snippet_iterator_t other_;
    edges_t::iterator backlink_;
  };

  void snippet::bind(snippet_iterator_t self,
                     snippet_iterator_t other,
                     kpm::vote const& vote) {
    auto e1{add(true, vote, other)};
    auto e2{other->add(false, vote, self)};

    e1->backlink_ = e2;
    e2->backlink_ = e1;
  }

  void snippet::unbind() {
    for (auto& edge : edges_) {
      edge.other_->edges_.erase(edge.backlink_);
    }
  }

  edges_t::iterator snippet::add(bool primary,
                                 kpm::vote const& vote,
                                 snippet_iterator_t other) {
    return edges_.emplace(edges_.end(), primary, vote, other);
  }

  [[nodiscard]] snippet extract_single(fgm::fragment&& fragment) {
    auto [image, mask]{fragment.blend()};

    sid::nat::dimg_t median{image.dimensions(), image.get_allocator()};

    kpe::extractor<grid_t, 0> extractor{image.dimensions()};
    return {std::move(fragment),
            std::move(mask),
            extractor.extract(image, median, image.get_allocator())};
  }

  template<typename Iter>
  [[nodiscard]] auto extract_all(Iter first, Iter last) {
    std::list<details::snippet> snippets{
        static_cast<std::size_t>(std::distance(first, last)),
        details::snippet{}};

    std::transform(
        std::execution::par, first, last, snippets.begin(), [](auto& fragment) {
          return extract_single(std::move(fragment));
        });

    return snippets;
  }

  struct match_config {
    using allocator_type = std::allocator<char>;

    static constexpr std::size_t weight_switch{
        std::numeric_limits<std::size_t>::max()};
    static constexpr std::size_t region_votes{3};

    [[nodiscard]] inline allocator_type get_allocator() const noexcept {
      return allocator_;
    }

    [[no_unique_address]] allocator_type allocator_;
  };

  template<typename It>
  void match_partial(It head, It first, It last) {
    for (; first != last; ++first) {
      if (auto ticket{kpm::match(details::match_config{},
                                 head->grid_[0],
                                 head->mask_,
                                 first->grid_[0],
                                 first->mask_)};
          !ticket.empty()) {
        head->bind(head, first, ticket.front());
      }
    }
  }

  template<typename It>
  void match_all(It first, It last) {
    for (auto rest{next(first)}; rest != last; ++first, ++rest) {
      match_partial(first, rest, last);
    }
  }
} // namespace details

template<typename Iter>
[[nodiscard]] std::vector<fgm::fragment> splice(Iter first, Iter last) {
  auto snippets{details::extract_all(first, last)};
  details::match_all(snippets.begin(), snippets.end());

  while (true) {
    std::vector<std::tuple<details::snippet_iterator_t, details::delta*>>
        deltas;
    for (auto s{snippets.begin()}; s != snippets.end(); ++s) {
      for (auto& e : s->edges_) {
        if (e.primary_) {
          deltas.emplace_back(s, &e);
        }
      }
    }

    if (deltas.empty()) {
      break;
    }

    auto& [left, edge]{*std::max_element(
        deltas.begin(), deltas.end(), [](auto& lhs, auto& rhs) {
          return std::get<1>(lhs)->vote_.count_ <
                 std::get<1>(rhs)->vote_.count_;
        })};

    auto right{edge->other_};

    auto& dst{left->fragment_};
    dst.blit(dst.zero() + edge->vote_.offset_, std::move(right->fragment_));
    snippets.emplace_front(details::extract_single(std::move(dst)));

    right->unbind();
    left->unbind();

    snippets.erase(right);
    snippets.erase(left);

    details::match_partial(
        snippets.begin(), std::next(snippets.begin()), snippets.end());
  }

  std::vector<fgm::fragment> result{};
  result.reserve(snippets.size());

  std::transform(snippets.begin(),
                 snippets.end(),
                 std::back_inserter(result),
                 [](auto& s) { return std::move(s.fragment_); });

  return result;
}

} // namespace fgs
