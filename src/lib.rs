#![doc = include_str!("../README.md")]

mod sumtree;

use std::{
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    ops::{Add, RangeInclusive},
};

use get_size2::GetSize;

pub use sumtree::Min;
use sumtree::{Arena, Bias, Item, SumTree};

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

impl<T: Copy> Item for RangeCount<T> {
    type Summary = CountSummary<T>;

    fn summary(&self) -> Self::Summary {
        CountSummary {
            end: self.end,
            count: self.count,
            min_count: self.count,
            entries: 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key<K>(K);

impl<K: Min> Min for Key<K> {
    const MIN: Self = Self(K::MIN);
}

impl<K> Add<CountSummary<K>> for Key<K> {
    type Output = Self;
    fn add(self, rhs: CountSummary<K>) -> Self::Output {
        Key(rhs.end)
    }
}

impl<T: Min> Min for CountSummary<T> {
    const MIN: Self = CountSummary {
        end: T::MIN,
        count: 0,
        min_count: usize::MAX,
        entries: 0,
    };
}

impl<T: Min> Add for CountSummary<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            entries: self.entries + rhs.entries,
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Tree<'a, T: Copy> {
            arena: &'a Arena<RangeCount<T>>,
            tree: &'a SumTree<RangeCount<T>>,
        }
        impl<T: fmt::Debug + Copy> fmt::Debug for Tree<'_, T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

impl<T: Min + Copy + Ord> CountRangeSketch<T> {
    pub fn new(limit: usize) -> Self {
        let mut arena = Arena::default();
        CountRangeSketch {
            tree: SumTree::new(&mut arena),
            arena,
            limit,
        }
    }

    pub fn reset(&mut self)
    where
        T: std::fmt::Debug,
    {
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

        while self.tree.summary.entries > self.limit {
            replace_with::replace_with_or_abort(&mut self.tree, |tree| {
                compact(tree, self.limit * 3 / 4, &mut self.arena)
            });
        }

        output
    }

    pub fn get_count(&mut self, t: T) -> (RangeInclusive<T>, usize) {
        let mut cursor = self.tree.cursor::<Key<T>>(&self.arena);
        cursor.seek_forward(&Key(t), Bias::Left, &self.arena);

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
        let start = self
            .tree
            .first(&self.arena)
            .map_or(T::MIN, |item| item.start);
        (start..=end, count)
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
    if tree.summary.entries <= limit {
        return tree;
    }

    if limit <= 1 {
        let CountSummary { count, end, .. } = tree.summary;
        let start = tree.first(arena).unwrap().start;
        arena.drop(tree);
        return SumTree::from_item(RangeCount { count, start, end }, arena);
    }

    let midpoint = mid_count_range(&tree, arena);

    let mut cursor = tree.into_cursor::<Key<T>>(arena);
    let mut left = cursor.slice(&Key(midpoint.end), Bias::Left, arena);

    if left.summary.entries == 0 {
        let left_item = cursor.next(arena).unwrap();
        let right = compact(cursor.suffix(arena), limit - 1, arena);
        return SumTree::from_item(left_item, arena).append(right, arena);
    }

    let left_limit = limit.div_ceil(2);
    left = compact(left, left_limit, arena);

    let right_limit = limit.saturating_sub(left.summary.entries);
    let right = compact(cursor.suffix(arena), right_limit, arena);

    let left_limit = limit.saturating_sub(right.summary.entries);
    left = compact(left, left_limit, arena);

    left.append(right, arena)
}

fn mid_count_range<T: Copy + Min>(
    tree: &SumTree<RangeCount<T>>,
    arena: &Arena<RangeCount<T>>,
) -> CountSummary<T> {
    let summary = tree.summary;
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
        .map_or(CountSummary::MIN, |item| item.summary())
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

    #[test]
    fn regression() {
        check_sketch_is_accurate(vec![
            323, 730, 746, 579, 352, 839, 598, 479, 552, 867, 472, 500, 523, 431, 593, 502, 615,
            717, 893, 399, 715, 771, 526, 797, 891, 405, 366, 496, 490, 800, 666, 689, 712, 485,
            882, 641, 819, 504, 880, 756, 626, 896, 413, 619, 678, 705, 325, 587, 839, 435, 572,
            443, 602, 807, 402, 545, 371, 777, 528, 742, 872, 512, 724, 833, 701, 791, 869, 444,
            362, 741, 561, 528, 797, 370, 820, 636, 673, 650, 559, 346, 625, 477, 336, 856, 336,
            360, 858, 388, 843, 795, 829, 484, 762, 447, 867, 590, 767, 445, 878, 766, 413, 533,
            485, 721, 875, 527, 312, 422, 701, 475, 762, 793, 761, 384, 466, 772, 762, 518, 365,
            713, 815, 874, 781, 718, 600, 879, 499, 558, 786, 707, 418, 779, 569, 557, 580, 581,
            813, 316, 723, 669, 585, 529, 512, 806, 632, 486, 573, 775, 335, 669, 847, 715, 653,
            548, 375, 723, 476, 639, 764, 485, 876, 781, 853, 616, 897, 713, 843, 658, 690, 779,
            846, 351, 700, 784, 847, 546, 625, 644, 708, 867, 553, 717, 497, 723, 524, 550, 688,
            636, 805, 474, 449, 687, 374, 402, 808, 354, 787, 646, 450, 354, 811, 705, 794, 660,
            441, 481, 730, 651, 779, 824, 613, 388, 651, 817, 501, 361, 683, 876, 748, 613, 472,
            313, 413, 323, 765, 462, 629, 367, 810, 648, 766, 561, 382, 626, 714, 472, 503, 611,
            875, 894, 531, 843, 800, 430, 447, 860, 722, 503, 652, 781, 694, 459, 758, 647, 688,
            674, 699, 505, 320, 549, 502, 630, 534, 331, 811, 555, 609, 653, 815, 369, 453, 438,
            633, 305, 646, 551, 344, 343, 666, 479, 328, 607, 714, 743, 355, 407,
        ]);
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
        assert!(size <= 36 * LIMIT, "heap size = {size}");
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
        height: 1,
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
                ],
            },
            Leaf {
                items: [
                    RangeCount {
                        range: 4..=4,
                        count: 4,
                    },
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
}
