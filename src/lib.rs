use std::{
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    ops::RangeInclusive,
};

use crate::sumtree::{Arena, Bias, Dimension, Item, SumTree, Summary};

mod sumtree;

#[derive(Clone, Copy)]
struct RangeCount<T> {
    count: usize,
    start: T,
    end: T,
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
    entries: usize,
    count: usize,
    min_count: usize,
    end: T,
}

impl<T: fmt::Debug> fmt::Debug for CountSummary<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_struct("CountSummary");
        f.field("range", &(..=&self.end));
        f.field("count", &self.count)
            .field("len", &self.entries)
            .finish()
    }
}

impl<T: Ranged> Item for RangeCount<T> {
    type Summary = CountSummary<T>;

    fn summary(&self, _: &()) -> Self::Summary {
        CountSummary {
            end: self.end,
            count: self.count,
            min_count: self.count,
            entries: 1,
        }
    }
}

pub trait Ranged: Ord + Copy {
    const MIN: Self;
}

impl<'a, K: Ranged> Dimension<'a, CountSummary<K>> for K {
    fn zero(_: &()) -> Self {
        K::MIN
    }

    fn add_summary(&mut self, summary: &'a CountSummary<K>, _: &()) {
        *self = summary.end;
    }
}

impl<T: Ranged> Summary for CountSummary<T> {
    type Context = ();

    fn zero(_: &()) -> Self {
        CountSummary {
            end: T::MIN,
            count: 0,
            min_count: usize::MAX,
            entries: 0,
        }
    }

    fn add_summary(&mut self, summary: &Self, _: &()) {
        self.end = summary.end;
        self.count += summary.count;
        self.min_count = std::cmp::min(self.min_count, summary.min_count);
        self.entries += summary.entries;
    }
}

pub struct CountRangeSketch<T: Ranged> {
    arena: Arena<RangeCount<T>>,
    tree: SumTree<RangeCount<T>>,
    limit: usize,
    cx: (),
}

impl<T: Ranged> CountRangeSketch<T> {
    pub fn new(limit: usize) -> Self {
        let cx = ();
        CountRangeSketch {
            arena: Arena::default(),
            tree: SumTree::new(&cx),
            limit,
            cx,
        }
    }

    pub fn count(&mut self, t: T) -> (RangeInclusive<T>, usize) {
        let cx = &self.cx;

        let output;
        self.tree = {
            let mut cursor = self.tree.cursor::<T>(cx);
            let new_tree = cursor.slice(&t, Bias::Left, &mut self.arena);

            let item = if let Some(cursor_item) = cursor.item(&self.arena)
                && t >= cursor_item.start
            {
                cursor.next();

                let mut item = *cursor_item;
                item.count += 1;
                item
            } else {
                RangeCount {
                    count: 1,
                    start: t,
                    end: t,
                }
            };
            output = (item.start..=item.end, item.count);

            new_tree.push(item, cx, &mut self.arena).append(
                cursor.suffix(&mut self.arena),
                cx,
                &mut self.arena,
            )
        };

        while self.tree.summary().entries > self.limit {
            self.tree = compact(
                std::mem::take(&mut self.tree),
                self.limit * 3 / 4,
                &self.cx,
                &mut self.arena,
            );
        }

        output
    }

    pub fn get_count(&mut self, t: T) -> (RangeInclusive<T>, usize) {
        let cx = &self.cx;

        let mut cursor = self.tree.cursor::<T>(cx);
        cursor.slice(&t, Bias::Left, &mut self.arena);

        if let Some(item) = cursor.item(&self.arena)
            && t >= item.start
        {
            (item.start..=item.end, item.count)
        } else {
            (t..=t, 0)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.tree.summary().entries
    }

    pub fn full(&self) -> (RangeInclusive<T>, usize) {
        let CountSummary { count, end, .. } = *self.tree.summary();
        let start = self.tree.first().map_or(T::MIN, |item| item.start);
        (start..=end, count)
    }

    pub fn get_all(&self) -> Vec<(RangeInclusive<T>, usize)> {
        self.tree
            .items(&self.cx, &self.arena)
            .into_iter()
            .map(|item| ((item.start..=item.end), item.count))
            .collect()
    }
}

fn compact<T: Ranged>(
    tree: SumTree<RangeCount<T>>,
    limit: usize,
    cx: &(),
    arena: &mut Arena<RangeCount<T>>,
) -> SumTree<RangeCount<T>> {
    if tree.summary().entries <= limit {
        return tree;
    }

    if limit <= 1 {
        let CountSummary { count, end, .. } = *tree.summary();
        let start = tree.first().unwrap().start;
        return SumTree::from_item(RangeCount { count, start, end }, cx);
    }

    let midpoint = mid_count_range(&tree, arena);

    let mut cursor = tree.cursor::<T>(cx);
    let mut left = cursor.slice(&midpoint.end, Bias::Left, arena);
    let mut right = cursor.suffix(arena);
    drop(cursor);

    if left.summary().entries == 0 || right.summary().entries == 0 {
        return tree;
    }

    let left_limit = limit.div_ceil(2);
    left = compact(left, left_limit, cx, arena);

    let right_limit = limit.saturating_sub(left.summary().entries);
    right = compact(right, right_limit, cx, arena);

    let left_limit = limit.saturating_sub(right.summary().entries);
    left = compact(left, left_limit, cx, arena);

    left.append(right, cx, arena)
}

fn mid_count_range<T: Ranged>(
    tree: &SumTree<RangeCount<T>>,
    arena: &Arena<RangeCount<T>>,
) -> CountSummary<T> {
    let summary = tree.summary();
    let mut cursor = tree.cursor::<CountSummary<T>>(&());

    let mut search = 0;
    cursor.search_forward(|_, child| {
        if summary.count.div_ceil(2) < search + child.count {
            true
        } else {
            search += child.count;
            false
        }
    });

    cursor
        .item(arena)
        .map_or_else(|| Summary::zero(&()), |item| item.summary(&()))
}

impl Ranged for u64 {
    const MIN: Self = Self::MIN;
}
impl Ranged for IpAddr {
    const MIN: IpAddr = IpAddr::V4(Ipv4Addr::from_bits(0));
}
impl Ranged for Ipv4Addr {
    const MIN: Ipv4Addr = Ipv4Addr::from_bits(0);
}
impl Ranged for Ipv6Addr {
    const MIN: Ipv6Addr = Ipv6Addr::from_bits(0);
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, ops::Range};

    use crate::CountRangeSketch;

    const LIMIT: usize = 20;
    const RANGE: Range<u64> = 30..90;
    const N: usize = 1000;

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn sketch_is_accurate(s in proptest::collection::vec(RANGE, N..=N)) {
            let mut actual = BTreeMap::<u64, usize>::new();
            let mut sketch = CountRangeSketch::new(LIMIT);

            for &n in &s {
                *actual.entry(n).or_default() += 1;

                let (range, count) = sketch.count(n);
                let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
                assert_eq!(actual_count, count);

                assert!(sketch.len() <= LIMIT);
            }

            for n in RANGE {
                let (range, count) = sketch.get_count(n);
                let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
                assert_eq!(actual_count, count);
            }

            let (range, count) = sketch.full();
            assert_eq!(count, s.len());

            let actual_count: usize = actual.range(range).map(|(_, v)| *v).sum();
            assert_eq!(actual_count, count);

        }
    }
}
