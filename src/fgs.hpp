
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
    [[nodiscard]] void bind(snippet_iterator_t self,
                            snippet_iterator_t other,
                            kpm::vote const& vote);

    void unbind();

    [[nodiscard]] edges_t::iterator
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

  inline void snippet::bind(snippet_iterator_t self,
                            snippet_iterator_t other,
                            kpm::vote const& vote) {
    auto edge1{add(true, vote, other)};
    auto edge2{other->add(false, vote, self)};

    edge1->backlink_ = edge2;
    edge2->backlink_ = edge1;
  }

  inline void snippet::unbind() {
    for (auto& edge : edges_) {
      edge.other_->edges_.erase(edge.backlink_);
    }
  }

  inline edges_t::iterator snippet::add(bool primary,
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

  [[nodiscard]] auto select_match(std::list<snippet>& snippets) {
    using item_t = std::tuple<snippet_iterator_t, delta*>;
    using result_t = std::optional<item_t>;

    std::vector<item_t> matches;
    for (auto it{snippets.begin()}; it != snippets.end(); ++it) {
      for (auto& edge : it->edges_) {
        if (edge.primary_) {
          matches.emplace_back(it, &edge);
        }
      }
    }

    if (!matches.empty()) {
      return result_t{*max_element(
          matches.begin(), matches.end(), [](auto& lhs, auto& rhs) {
            return get<1>(lhs)->vote_.count_ < get<1>(rhs)->vote_.count_;
          })};
    }

    return result_t{};
  }

  void splice_single(std::list<snippet>& snippets,
                     snippet_iterator_t left,
                     delta* edge) {
    auto right{edge->other_};

    auto& dst{left->fragment_};
    dst.blit(dst.zero() + edge->vote_.offset_, std::move(right->fragment_));
    dst.normalize();

    snippets.emplace_front(extract_single(std::move(dst)));

    right->unbind();
    left->unbind();

    snippets.erase(right);
    snippets.erase(left);

    match_partial(snippets.begin(), next(snippets.begin()), snippets.end());
  }

} // namespace details

template<typename Iter>
[[nodiscard]] std::vector<fgm::fragment> splice(Iter first, Iter last) {
  using namespace details;

  auto snippets{extract_all(first, last)};
  match_all(snippets.begin(), snippets.end());

  while (true) {
    auto match{select_match(snippets)};
    if (!match) {
      break;
    }

    auto& [left, edge]{*match};
    splice_single(snippets, left, edge);
  }

  std::vector<fgm::fragment> result{};
  result.reserve(snippets.size());

  transform(snippets.begin(),
            snippets.end(),
            back_inserter(result),
            [](auto& s) { return std::move(s.fragment_); });

  return result;
}

} // namespace fgs
