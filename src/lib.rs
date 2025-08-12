use std::{
    fmt::{self, Debug},
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    ops::RangeInclusive,
};

use get_size2::GetSize;

use crate::sumtree::{Arena, Bias, Dimension, Item, SumTree, Summary};

mod sumtree;

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
    entries: usize,
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

pub trait Ranged: Ord + Copy + Debug {
    const MIN: Self;
}

impl<K: Ranged> Dimension<CountSummary<K>> for K {
    fn zero(_: &()) -> Self {
        K::MIN
    }

    fn add_summary(&mut self, summary: CountSummary<K>, _: &()) {
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

    fn add_summary(&mut self, summary: Self, _: &()) {
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
        let mut arena = Arena::default();
        CountRangeSketch {
            tree: SumTree::new(&cx, &mut arena),
            arena,
            limit,
            cx,
        }
    }

    pub fn reset(&mut self) {
        replace_with::replace_with_or_abort(&mut self.tree, |tree| {
            if cfg!(debug_assertions) {
                self.arena.drop(tree);
                assert!(self.arena.is_empty());
            }
            self.arena.clear();
            SumTree::new(&self.cx, &mut self.arena)
        });
    }

    pub fn count(&mut self, t: T) -> (RangeInclusive<T>, usize) {
        let cx = &self.cx;
        let arena = &mut self.arena;

        let mut output = (t..=t, 1);
        replace_with::replace_with_or_abort(&mut self.tree, |tree| {
            let mut cursor = tree.into_cursor::<T>(cx);
            let mut new_tree = cursor.slice(&t, Bias::Left, arena);

            let new_item = RangeCount {
                count: 1,
                start: t,
                end: t,
            };
            match cursor.next(arena) {
                Some(mut item) if t >= item.start => {
                    item.count += 1;
                    output = (item.start..=item.end, item.count);
                    new_tree = new_tree.push(item, cx, arena)
                }
                Some(item) => {
                    new_tree = new_tree.push(new_item, cx, arena).push(item, cx, arena);
                }
                None => {
                    new_tree = new_tree.push(new_item, cx, arena);
                }
            };

            new_tree.append(cursor.suffix(arena), cx, arena)
        });

        while self.tree.summary.entries > self.limit {
            replace_with::replace_with_or_abort(&mut self.tree, |tree| {
                compact(tree, self.limit * 3 / 4, &self.cx, arena)
            });
        }

        output
    }

    pub fn get_count(&mut self, t: T) -> (RangeInclusive<T>, usize) {
        let cx = &self.cx;

        let mut cursor = self.tree.cursor::<T>(cx);
        cursor.seek_forward(&t, Bias::Left, &self.arena);

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
        self.tree.summary.entries
    }

    pub fn full(&self) -> (RangeInclusive<T>, usize) {
        let CountSummary { count, end, .. } = self.tree.summary;
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
    if tree.summary.entries <= limit {
        return tree;
    }

    if limit <= 1 {
        let CountSummary { count, end, .. } = tree.summary;
        let start = tree.first().unwrap().start;
        arena.drop(tree);
        return SumTree::from_item(RangeCount { count, start, end }, cx, arena);
    }

    let midpoint = mid_count_range(&tree, arena);

    let mut cursor = tree.into_cursor::<T>(cx);
    let mut left = cursor.slice(&midpoint.end, Bias::Left, arena);

    if left.summary.entries == 0 {
        let left_item = cursor.next(arena).unwrap();
        let right = compact(cursor.suffix(arena), limit - 1, cx, arena);
        return SumTree::from_item(left_item, cx, arena).append(right, cx, arena);
    }

    let left_limit = limit.div_ceil(2);
    left = compact(left, left_limit, cx, arena);

    let right_limit = limit.saturating_sub(left.summary.entries);
    let right = compact(cursor.suffix(arena), right_limit, cx, arena);

    let left_limit = limit.saturating_sub(right.summary.entries);
    left = compact(left, left_limit, cx, arena);

    left.append(right, cx, arena)
}

fn mid_count_range<T: Ranged>(
    tree: &SumTree<RangeCount<T>>,
    arena: &Arena<RangeCount<T>>,
) -> CountSummary<T> {
    let summary = tree.summary;
    let mut cursor = tree.cursor::<CountSummary<T>>(&());

    let mut search = 0;
    cursor.search_forward(|child| {
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

        // reset
        actual.clear();
        sketch.reset();

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

        let size = sketch.tree.get_heap_size() + sketch.arena.get_heap_size();
        assert!(size <= 128 * LIMIT, "heap size = {size}");
    }
}
