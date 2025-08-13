#![doc = include_str!("../README.md")]

mod sumtree;

use std::{
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    ops::{Add, RangeBounds, RangeInclusive},
};

use get_size2::GetSize;
use sumtree::{Arena, Bias, Item, SumTree};

/// A type that has a minimum value.
pub trait Min: Ord {
    /// The minimum value for the type.
    ///
    /// Invariant: `T::MIN <= t` must be true for all `t`.
    const MIN: Self;
}

#[derive(Clone, Copy)]
struct RangeCount<T> {
    count: usize,
    start: T,
    end: T,
}

impl<T: GetSize> GetSize for RangeCount<T> {
    fn get_heap_size(&self) -> usize {
        self.start.get_heap_size() + self.end.get_heap_size()
    }
}

impl<T: fmt::Debug> fmt::Debug for RangeCount<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RangeCount")
            .field("range", &(&self.start..=&self.end))
            .field("count", &self.count)
            .finish()
    }
}

#[derive(Clone, Copy)]
struct CountSummary<T> {
    len: usize,
    count: usize,
    min_count: usize,
    end: T,
}

impl<T: GetSize> GetSize for CountSummary<T> {
    fn get_heap_size(&self) -> usize {
        self.end.get_heap_size()
    }
}

impl<T: fmt::Debug> fmt::Debug for CountSummary<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountSummary")
            .field("range", &(..=&self.end))
            .field("count", &self.count)
            .field("len", &self.len)
            .finish()
    }
}

impl<T: Copy> Item for RangeCount<T> {
    type Summary = CountSummary<T>;

    fn summary(&self) -> Self::Summary {
        CountSummary {
            end: self.end,
            count: self.count,
            min_count: self.count,
            len: 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key<K>(K);

impl<K: Min> sumtree::Min for Key<K> {
    const MIN: Self = Self(K::MIN);
}

// impl<K: Ord> Comparable<Key<K>> for K {
//     fn compare(&self, key: &Key<K>) -> std::cmp::Ordering {
//         self.cmp(&key.0)
//     }
// }

impl<K> Add<CountSummary<K>> for Key<K> {
    type Output = Self;
    fn add(self, rhs: CountSummary<K>) -> Self::Output {
        Key(rhs.end)
    }
}

impl<T: Min> sumtree::Min for CountSummary<T> {
    const MIN: Self = CountSummary {
        end: T::MIN,
        count: 0,
        min_count: usize::MAX,
        len: 0,
    };
}

impl<T: Min> Add for CountSummary<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            len: self.len + rhs.len,
            count: self.count + rhs.count,
            min_count: std::cmp::min(self.min_count, rhs.min_count),
            end: rhs.end,
        }
    }
}

pub struct CountRangeSketch<T: Copy> {
    arena: Arena<RangeCount<T>>,
    tree: SumTree<RangeCount<T>>,
    limit: usize,
}

impl<T: fmt::Debug + Copy> fmt::Debug for CountRangeSketch<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Tree<'a, T: Copy> {
            arena: &'a Arena<RangeCount<T>>,
            tree: &'a SumTree<RangeCount<T>>,
        }
        impl<T: fmt::Debug + Copy> fmt::Debug for Tree<'_, T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.tree.fmt(f, self.arena)
            }
        }

        f.debug_struct("CountRangeSketch")
            .field(
                "tree",
                &Tree {
                    arena: &self.arena,
                    tree: &self.tree,
                },
            )
            .field("limit", &self.limit)
            .finish()
    }
}

impl<T: Min + Copy> CountRangeSketch<T> {
    pub fn new(limit: usize) -> Self {
        let mut arena = Arena::default();
        CountRangeSketch {
            tree: SumTree::new(&mut arena),
            arena,
            limit,
        }
    }

    pub fn reset(&mut self) {
        self.arena.reset(&mut self.tree);
    }

    pub fn count(&mut self, t: T) -> (RangeInclusive<T>, usize) {
        let arena = &mut self.arena;

        let mut output = (t..=t, 1);
        replace_with::replace_with_or_abort(&mut self.tree, |tree| {
            let mut cursor = tree.into_cursor::<Key<T>>(arena);
            let mut new_tree = cursor.slice(&Key(t), Bias::Left, arena);

            let new_item = RangeCount {
                count: 1,
                start: t,
                end: t,
            };
            match cursor.next(arena) {
                Some(mut item) if t >= item.start => {
                    item.count += 1;
                    output = (item.start..=item.end, item.count);
                    new_tree = new_tree.push(item, arena)
                }
                Some(item) => {
                    new_tree = new_tree.push(new_item, arena).push(item, arena);
                }
                None => {
                    new_tree = new_tree.push(new_item, arena);
                }
            };

            new_tree.append(cursor.suffix(arena), arena)
        });

        if self.tree.summary().len > self.limit {
            // compact to 75% of the current size to amortise the compaction cost.
            let new_limit = (self.limit * 3).div_ceil(4);
            replace_with::replace_with_or_abort(&mut self.tree, |tree| {
                compact(tree, new_limit, arena)
            });
            debug_assert!(self.len() <= new_limit);
        }

        output
    }

    /// Return the number of occurences for the smallest range that fully contains the requested range.
    pub fn get(&self, range: impl RangeBounds<T>) -> (RangeInclusive<T>, usize) {
        let arena = &self.arena;
        let mut cursor = self.tree.cursor::<Key<T>>(arena);

        let (start_bound, start_inclusive) = match range.start_bound() {
            std::ops::Bound::Included(x) => (*x, true),
            std::ops::Bound::Excluded(x) => (*x, false),
            std::ops::Bound::Unbounded => (T::MIN, true),
        };

        let (end_bound, end_inclusive) = match range.end_bound() {
            std::ops::Bound::Included(x) => (*x, true),
            std::ops::Bound::Excluded(x) => (*x, false),
            std::ops::Bound::Unbounded => (self.tree.summary().end, true),
        };

        cursor.seek_forward(&Key(start_bound), Bias::Left, arena);
        if !start_inclusive
            && let Some(&start) = cursor.item(arena)
            && start_bound == start.end
        {
            // skip.
            cursor.next(arena);
        }

        let Some(&start) = cursor.item(arena) else {
            return (start_bound..=end_bound, 0);
        };

        if end_bound < start.start || (!end_inclusive && end_bound == start.start) {
            return (start_bound..=end_bound, 0);
        }

        let mut summary: CountSummary<T> = cursor.summary(&Key(end_bound), Bias::Left, arena);

        if let Some(end) = cursor.item(arena) {
            match std::cmp::Ord::cmp(&end.start, &end_bound) {
                std::cmp::Ordering::Greater => {}
                std::cmp::Ordering::Equal if !end_inclusive => {}
                // The next item overlaps with our requested range.
                _ => summary = summary + end.summary(),
            }
        }

        (start.start..=summary.end, summary.count)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.tree.summary().len
    }

    pub fn get_all(&self) -> Vec<(RangeInclusive<T>, usize)> {
        self.tree
            .items(&self.arena)
            .into_iter()
            .map(|item| ((item.start..=item.end), item.count))
            .collect()
    }
}

fn compact<T: Copy + Min + Ord>(
    tree: SumTree<RangeCount<T>>,
    limit: usize,
    arena: &mut Arena<RangeCount<T>>,
) -> SumTree<RangeCount<T>> {
    if tree.summary().len <= limit {
        return tree;
    }

    if limit <= 1 {
        let CountSummary { count, end, .. } = *tree.summary();
        let start = tree.first(arena).unwrap().start;
        arena.drop(tree);
        return SumTree::from_item(RangeCount { count, start, end }, arena);
    }

    let midpoint = mid_count_range(&tree, arena);

    let mut cursor = tree.into_cursor::<Key<T>>(arena);
    let mut left = cursor.slice(&Key(midpoint.end), Bias::Left, arena);

    if left.summary().len == 0 {
        arena.drop(left);
        let left_item = cursor.next(arena).unwrap();
        let right = compact(cursor.suffix(arena), limit - 1, arena);
        return SumTree::from_item(left_item, arena).append(right, arena);
    }

    let left_limit = limit.div_ceil(2);
    left = compact(left, left_limit, arena);

    let right_limit = limit.saturating_sub(left.summary().len);
    let right = compact(cursor.suffix(arena), right_limit, arena);

    let left_limit = limit.saturating_sub(right.summary().len);
    left = compact(left, left_limit, arena);

    left.append(right, arena)
}

fn mid_count_range<T: Copy + Min>(
    tree: &SumTree<RangeCount<T>>,
    arena: &Arena<RangeCount<T>>,
) -> CountSummary<T> {
    let summary = *tree.summary();
    let mut cursor = tree.cursor::<CountSummary<T>>(arena);

    let mut search = 0;
    cursor.search_forward(
        |child| {
            if summary.count.div_ceil(2) < search + child.count {
                true
            } else {
                search += child.count;
                false
            }
        },
        arena,
    );

    cursor
        .item(arena)
        .map_or(sumtree::Min::MIN, |item| item.summary())
}

impl Min for u64 {
    const MIN: Self = Self::MIN;
}
impl Min for IpAddr {
    const MIN: IpAddr = IpAddr::V4(Ipv4Addr::from_bits(0));
}
impl Min for Ipv4Addr {
    const MIN: Ipv4Addr = Ipv4Addr::from_bits(0);
}
impl Min for Ipv6Addr {
    const MIN: Ipv6Addr = Ipv6Addr::from_bits(0);
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, ops::Range};

    use crate::CountRangeSketch;

    const LIMIT: usize = 100;
    const RANGE: Range<u64> = 300..900;

    use get_size2::GetSize;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn sketch_is_accurate(s in proptest::collection::vec(RANGE, 10..=5000)) {
            check_sketch_is_accurate(s);
        }
    }

    fn check_sketch_is_accurate(s: Vec<u64>) {
        let mut actual = BTreeMap::<u64, usize>::new();
        let mut sketch = CountRangeSketch::new(LIMIT);

        for &n in &s {
            *actual.entry(n).or_default() += 1;

            let (range, count) = sketch.count(n);
            let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
            assert_eq!(actual_count, count);

            assert!(sketch.len() <= LIMIT);
            sketch.arena.assert_reachability(&sketch.tree);
        }

        for n in RANGE {
            let (range, count) = sketch.get(n..=n);
            let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
            assert_eq!(actual_count, count);
        }

        let (range, count) = sketch.get(..);
        assert_eq!(count, s.len());

        let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
        assert_eq!(actual_count, count);

        // reset
        actual.clear();
        sketch.reset();
        sketch.arena.assert_reachability(&sketch.tree);

        for &n in &s {
            *actual.entry(n).or_default() += 1;

            let (range, count) = sketch.count(n);
            let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
            assert_eq!(actual_count, count);

            assert!(sketch.len() <= LIMIT);
        }

        for n in RANGE {
            let (range, count) = sketch.get(n..=n);
            let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
            assert_eq!(actual_count, count);
        }

        let (range, count) = sketch.get(..);
        assert_eq!(count, s.len());

        let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
        assert_eq!(actual_count, count);

        let size = sketch.tree.get_heap_size() + sketch.arena.get_heap_size();
        assert!(size <= 20 * LIMIT, "heap size = {size}");
    }

    #[test]
    fn check_debug() {
        let mut sketch = CountRangeSketch::new(20);
        sketch.count(1);
        sketch.count(2);
        sketch.count(3);
        sketch.count(4);
        sketch.count(5);
        sketch.count(6);
        sketch.count(2);
        sketch.count(3);
        sketch.count(4);
        sketch.count(5);
        sketch.count(6);
        sketch.count(2);
        sketch.count(3);
        sketch.count(4);
        sketch.count(5);
        sketch.count(3);
        sketch.count(4);

        let expected = "CountRangeSketch {
    tree: Internal {
        summary: CountSummary {
            range: ..=6,
            count: 17,
            len: 6,
        },
        children: [
            Leaf {
                items: [
                    RangeCount {
                        range: 1..=1,
                        count: 1,
                    },
                    RangeCount {
                        range: 2..=2,
                        count: 3,
                    },
                    RangeCount {
                        range: 3..=3,
                        count: 4,
                    },
                    RangeCount {
                        range: 4..=4,
                        count: 4,
                    },
                ],
            },
            Leaf {
                items: [
                    RangeCount {
                        range: 5..=5,
                        count: 3,
                    },
                    RangeCount {
                        range: 6..=6,
                        count: 2,
                    },
                ],
            },
        ],
    },
    limit: 20,
}";
        assert_eq!(format!("{sketch:#?}"), expected);
    }

    #[test]
    fn ranges() {
        let mut sketch = CountRangeSketch::new(4);

        sketch.count(1);
        sketch.count(1);
        sketch.count(1);
        sketch.count(1);

        sketch.count(2);
        sketch.count(2);
        sketch.count(2);

        sketch.count(3);
        sketch.count(3);

        sketch.count(5);

        // All counts are as precise as they can be
        assert_eq!(sketch.get(1..=1), (1..=1, 4));
        assert_eq!(sketch.get(2..=2), (2..=2, 3));
        assert_eq!(sketch.get(3..=3), (3..=3, 2));
        assert_eq!(sketch.get(4..=4), (4..=4, 0));
        assert_eq!(sketch.get(5..=5), (5..=5, 1));

        // we can request different ranges
        assert_eq!(sketch.get(..3), (1..=2, 7));
        assert_eq!(sketch.get(3..), (3..=5, 3));
        assert_eq!(
            sketch.get((std::ops::Bound::Excluded(1), std::ops::Bound::Included(3),)),
            (2..=3, 5)
        );
        assert_eq!(sketch.get(..), (1..=5, 10));

        assert_eq!(sketch.len(), 4);

        // The counts will merge when exceeding the limit.
        sketch.count(4);
        assert_eq!(sketch.get(1..=1), (1..=1, 4));
        assert_eq!(sketch.get(2..=2), (2..=2, 3));
        assert_eq!(sketch.get(3..=3), (3..=5, 4));
        assert_eq!(sketch.get(4..=4), (3..=5, 4));
        assert_eq!(sketch.get(5..=5), (3..=5, 4));

        assert_eq!(sketch.get(..3), (1..=2, 7));
        assert_eq!(sketch.get(3..), (3..=5, 4));
        assert_eq!(sketch.get(2..4), (2..=5, 7));
        assert_eq!(sketch.get(..), (1..=5, 11));

        assert_eq!(sketch.len(), 3);
    }
}
