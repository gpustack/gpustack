"""Tolerant keyword matching for in-memory catalog search.

Model names are dense in separators and digits, and users type them
inconsistently: with spaces, with dashes, with underscores, with the size
first. So the query is split on whitespace and both sides are squashed to
alphanumerics, and a candidate matches when every token appears in it. Token
order is therefore irrelevant, and everything that scores a match is
order-insensitive too.

Squashing separators away makes short numeric tokens promiscuous -- a `35`
typed as `3.5` also occurs inside a `235B` -- so matches are scored and ranked
rather than merely filtered, and anything far below the best match is dropped.
Scoring rewards matches that land on the candidate's own unit boundaries, where
a unit ends at a separator *or* at a letter/digit transition. The second kind
matters because version numbers sit there: in a name like ``foo3.5-2b`` the
units are ``foo|3|5|2|b``, so a ``3.5`` lines up even though nothing separates
the ``3`` from the ``5`` once squashed.

No edit-distance fuzziness, deliberately. Model names differ from their
siblings by a single character, so typo tolerance would make neighbouring
releases and sizes match each other -- worse than not matching at all.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set, Tuple, TypeVar

T = TypeVar("T")

# Whole-candidate scores, awarded on top of the per-token scores below.
_EXACT = 40
"""The tokens account for the whole candidate."""
_CONTIGUOUS = 25
"""The tokens sit next to each other in the candidate, ignoring separators."""
_PREFIX = 15
"""...starting at its very beginning."""

# Per-token scores. Only the best occurrence of each token is scored. Every
# candidate in a result set matches every token, so the token count is
# constant and these sums stay comparable.
_ALIGNED = 10
"""The token spans whole units."""
_START_ALIGNED = 6
"""The token starts on a unit boundary."""
_END_ALIGNED = 3
"""The token ends on a unit boundary."""
_UNALIGNED = 1
"""The token appears mid-unit."""

_RELEVANCE_FLOOR = 0.4
"""On a multi-token query, drop candidates below this fraction of the best.

Ranking alone fixes what the first page shows; it leaves the result count and
every later page full of incidental matches. The bound is relative because
scores are only comparable within one query -- more tokens means a higher
ceiling, so no absolute cut-off transfers between queries.

It applies only to multi-token queries because that is where the noise comes
from: separators are dropped, so combining tokens starts matching across parts
of a name that have nothing to do with each other. A single token is an
ordinary contains-search, and its scores span a narrow range -- a genuine
mid-unit hit (a model whose family name embeds the query, say) sits at 30% of
an exact one and has to survive.
"""


def _squash(text: str) -> str:
    """Lowercase and drop everything that is not alphanumeric."""
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _normalize(text: str) -> Tuple[str, Set[int]]:
    """Squash `text` and locate its unit boundaries.

    Returns the squashed text and the offsets into it at which a unit starts.
    A unit ends at a separator or at a letter/digit transition.
    """
    squashed: List[str] = []
    starts: Set[int] = set()
    previous: Optional[str] = None  # None also means "a separator preceded this"
    for ch in text.casefold():
        if not ch.isalnum():
            previous = None
            continue
        if previous is None or ch.isdigit() != previous.isdigit():
            starts.add(len(squashed))
        squashed.append(ch)
        previous = ch
    return "".join(squashed), starts


@dataclass(frozen=True)
class _SearchQuery:
    """A parsed search query, scored against one candidate at a time."""

    tokens: Tuple[str, ...]
    """Squashed query tokens. Every one must match for a candidate to match."""

    def score(self, text: str) -> Optional[int]:
        """Score `text` against this query, or None if it does not match.

        Higher is better. Scores are only comparable within one query.
        """
        squashed, starts = _normalize(text)
        # A unit ends where the next one starts, or at the end of the text.
        ends = starts | {len(squashed)}

        total = 0
        placements = []
        for token in self.tokens:
            match = self._best_token_match(token, squashed, starts, ends)
            if match is None:
                return None
            score, position = match
            total += score
            placements.append((position, token))

        # Judged in the order the tokens occur in the candidate rather than the
        # order they were typed, so that two spellings of the same intent score
        # alike.
        placements.sort()
        run = "".join(token for _, token in placements)
        if run in squashed:
            # Adjacency is a claim about how the tokens sit relative to each
            # other, so it earns nothing when there is only one -- a lone token
            # is trivially contiguous, and paying for that would flatten the
            # gap between a real match and an incidental one.
            if len(self.tokens) > 1:
                total += _CONTIGUOUS
            if squashed.startswith(run):
                total += _PREFIX
            if squashed == run:
                total += _EXACT

        return total

    @staticmethod
    def _best_token_match(
        token: str, squashed: str, starts: Set[int], ends: Set[int]
    ) -> Optional[Tuple[int, int]]:
        """The best-aligned occurrence of `token` as (score, position)."""
        best = None
        position = squashed.find(token)
        while position != -1:
            end = position + len(token)
            if position in starts:
                score = _ALIGNED if end in ends else _START_ALIGNED
            else:
                score = _END_ALIGNED if end in ends else _UNALIGNED

            if best is None or score > best[0]:
                best = (score, position)
            if score == _ALIGNED:
                break

            position = squashed.find(token, position + 1)

        return best


def _compile_search_query(search: Optional[str]) -> Optional[_SearchQuery]:
    """Parse a raw search string, or return None if it carries no keywords.

    None means "no filtering": this is what makes a blank or punctuation-only
    query behave like no query at all, including one that is nothing but
    leading or trailing spaces.
    """
    if not search:
        return None

    tokens = tuple(token for token in map(_squash, search.split()) if token)
    if not tokens:
        return None

    return _SearchQuery(tokens=tokens)


def rank_matches(
    items: Sequence[T], search: Optional[str], key: Callable[[T], str]
) -> List[T]:
    """Keep the items relevant to `search`, best match first.

    Ties keep the caller's ordering, so a broad query leaves a curated
    sequence intact instead of reshuffling it by score noise.
    """
    query = _compile_search_query(search)
    if query is None:
        return list(items)

    scored = []
    for item in items:
        score = query.score(key(item))
        if score is not None:
            scored.append((score, item))

    if scored and len(query.tokens) > 1:
        floor = max(score for score, _ in scored) * _RELEVANCE_FLOOR
        scored = [pair for pair in scored if pair[0] >= floor]

    scored.sort(key=lambda pair: -pair[0])
    return [item for _, item in scored]
