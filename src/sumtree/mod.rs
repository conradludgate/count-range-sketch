mod cursor;

use arrayvec::ArrayVec;
pub use cursor::{Cursor, Iter};
use slotmap::SlotMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::{cmp::Ordering, fmt, iter::FromIterator, sync::Arc};

#[cfg(test)]
pub const TREE_BASE: usize = 2;
#[cfg(not(test))]
pub const TREE_BASE: usize = 6;

slotmap::new_key_type! {
    struct NodeKey;
}

pub struct Arena<T: Item>(SlotMap<NodeKey, Node<T>>);

impl<T: Item> Default for Arena<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// An item that can be stored in a [`SumTree`]
///
/// Must be summarized by a type that implements [`Summary`]
pub trait Item: Copy {
    type Summary: Summary;

    fn summary(&self, cx: &<Self::Summary as Summary>::Context) -> Self::Summary;
}

/// A type that describes the Sum of all [`Item`]s in a subtree of the [`SumTree`]
///
/// Each Summary type can have multiple [`Dimension`]s that it measures,
/// which can be used to navigate the tree
pub trait Summary: Copy {
    type Context;

    fn zero(cx: &Self::Context) -> Self;
    fn add_summary(&mut self, summary: &Self, cx: &Self::Context);
}

/// Each [`Summary`] type can have more than one [`Dimension`] type that it measures.
///
/// You can use dimensions to seek to a specific location in the [`SumTree`]
///
/// # Example:
/// Zed's rope has a `TextSummary` type that summarizes lines, characters, and bytes.
/// Each of these are different dimensions we may want to seek to
pub trait Dimension<'a, S: Summary>: Copy {
    fn zero(cx: &S::Context) -> Self;

    fn add_summary(&mut self, summary: &'a S, cx: &S::Context);
}

impl<'a, T: Summary> Dimension<'a, T> for T {
    fn zero(cx: &T::Context) -> Self {
        Summary::zero(cx)
    }

    fn add_summary(&mut self, summary: &'a T, cx: &T::Context) {
        Summary::add_summary(self, summary, cx);
    }
}

pub trait SeekTarget<'a, S: Summary, D: Dimension<'a, S>> {
    fn cmp(&self, cursor_location: &D, cx: &S::Context) -> Ordering;
}

impl<'a, S: Summary, D: Dimension<'a, S> + Ord> SeekTarget<'a, S, D> for D {
    fn cmp(&self, cursor_location: &Self, _: &S::Context) -> Ordering {
        Ord::cmp(self, cursor_location)
    }
}

impl<'a, T: Summary> Dimension<'a, T> for () {
    fn zero(_: &T::Context) -> Self {}

    fn add_summary(&mut self, _: &'a T, _: &T::Context) {}
}

/// Bias is used to settle ambiguities when determining positions in an ordered sequence.
///
/// The primary use case is for text, where Bias influences
/// which character an offset or anchor is associated with.
///
/// # Examples
/// Given the buffer `AˇBCD`:
/// - The offset of the cursor is 1
/// - [Bias::Left] would attach the cursor to the character `A`
/// - [Bias::Right] would attach the cursor to the character `B`
///
/// Given the buffer `A«BCˇ»D`:
/// - The offset of the cursor is 3, and the selection is from 1 to 3
/// - The left anchor of the selection has [Bias::Right], attaching it to the character `B`
/// - The right anchor of the selection has [Bias::Left], attaching it to the character `C`
///
/// Given the buffer `{ˇ<...>`, where `<...>` is a folded region:
/// - The display offset of the cursor is 1, but the offset in the buffer is determined by the bias
/// - [Bias::Left] would attach the cursor to the character `{`, with a buffer offset of 1
/// - [Bias::Right] would attach the cursor to the first character of the folded region,
///   and the buffer offset would be the offset of the first character of the folded region
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Debug, Hash, Default)]
pub enum Bias {
    /// Attach to the character on the left
    #[default]
    Left,
    /// Attach to the character on the right
    Right,
}

/// A B+ tree in which each leaf node contains `Item`s of type `T` and a `Summary`s for each `Item`.
/// Each internal node contains a `Summary` of the items in its subtree.
///
/// The maximum number of items per node is `TREE_BASE * 2`.
///
/// Any [`Dimension`] supported by the [`Summary`] type can be used to seek to a specific location in the tree.
#[derive(Clone)]
pub struct SumTree<T: Item>(Arc<Node<T>>, T::Summary);

// impl<T> fmt::Debug for SumTree<T>
// where
//     T: fmt::Debug + Item,
//     T::Summary: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         f.debug_tuple("SumTree").field(&self.0).finish()
//     }
// }

impl<T: Item> SumTree<T> {
    pub fn new(cx: &<T::Summary as Summary>::Context, arena: &mut Arena<T>) -> Self {
        SumTree::from_summary(<T::Summary as Summary>::zero(cx), arena)
    }

    /// Useful in cases where the item type has a non-trivial context type, but the zero value of the summary type doesn't depend on that context.
    pub fn from_summary(summary: T::Summary, arena: &mut Arena<T>) -> Self {
        SumTree(
            Arc::new(Node::Leaf {
                items: ArrayVec::new(),
                item_summaries: ArrayVec::new(),
            }),
            summary,
        )
    }

    pub fn from_item(item: T, cx: &<T::Summary as Summary>::Context) -> Self {
        let summary = item.summary(cx);
        SumTree(
            Arc::new(Node::Leaf {
                items: ArrayVec::from_iter(Some(item)),
                item_summaries: ArrayVec::from_iter(Some(summary)),
            }),
            summary,
        )
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }

    pub fn cursor<'a, S>(&'a self, cx: &'a <T::Summary as Summary>::Context) -> Cursor<'a, T, S>
    where
        S: Dimension<'a, T::Summary>,
    {
        Cursor::new(self, cx)
    }

    #[allow(dead_code)]
    pub fn first(&self) -> Option<&T> {
        self.leftmost_leaf().0.items().first()
    }

    // pub fn last(&self) -> Option<&T> {
    //     self.rightmost_leaf().0.items().last()
    // }

    #[cfg(test)]
    pub fn extent<'a, D: Dimension<'a, T::Summary>>(
        &'a self,
        cx: &<T::Summary as Summary>::Context,
    ) -> D {
        let mut extent = D::zero(cx);
        extent.add_summary(&self.1, cx);
        extent
    }

    pub fn summary(&self) -> &T::Summary {
        &self.1
    }

    pub fn is_empty(&self) -> bool {
        match self.0.as_ref() {
            Node::Internal { .. } => false,
            Node::Leaf { items, .. } => items.is_empty(),
        }
    }

    fn from_child_trees(
        left: SumTree<T>,
        right: SumTree<T>,
        cx: &<T::Summary as Summary>::Context,
    ) -> Self {
        let height = left.0.height() + 1;
        let mut child_summaries = ArrayVec::new();
        child_summaries.push(*left.summary());
        child_summaries.push(*right.summary());
        let mut child_trees = ArrayVec::new();
        child_trees.push(left);
        child_trees.push(right);
        SumTree {
            1: sum(child_summaries.iter(), cx),
            0: Arc::new(Node::Internal {
                height,
                child_summaries,
                child_trees,
            }),
        }
    }

    fn leftmost_leaf(&self) -> &Self {
        match *self.0 {
            Node::Leaf { .. } => self,
            Node::Internal {
                ref child_trees, ..
            } => child_trees.first().unwrap().leftmost_leaf(),
        }
    }

    // fn rightmost_leaf(&self) -> &Self {
    //     match *self.0 {
    //         Node::Leaf { .. } => self,
    //         Node::Internal {
    //             ref child_trees, ..
    //         } => child_trees.last().unwrap().rightmost_leaf(),
    //     }
    // }
}
impl<T: Item + Clone> SumTree<T> {
    pub fn items(&self, cx: &<T::Summary as Summary>::Context, arena: &Arena<T>) -> Vec<T> {
        let mut items = Vec::new();
        let mut cursor = self.cursor::<()>(cx);
        cursor.next();
        while let Some(item) = cursor.item(arena) {
            items.push(item.clone());
            cursor.next();
        }
        items
    }

    #[cfg(test)]
    pub fn extend<I>(
        self,
        iter: I,
        cx: &<T::Summary as Summary>::Context,
        arena: &mut Arena<T>,
    ) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        iter.into_iter()
            .fold(self, |this, i| this.push(i, cx, arena))
    }

    pub fn push(
        self,
        item: T,
        cx: &<T::Summary as Summary>::Context,
        arena: &mut Arena<T>,
    ) -> Self {
        self.append(Self::from_item(item, cx), cx, arena)
    }

    pub fn append(
        mut self,
        other: Self,
        cx: &<T::Summary as Summary>::Context,
        arena: &mut Arena<T>,
    ) -> Self {
        if self.is_empty() {
            return other;
        } else if !other.0.is_leaf() || !other.0.items().is_empty() {
            if self.0.height() < other.0.height() {
                for tree in other.0.child_trees() {
                    self = self.append(tree.clone(), cx, arena);
                }
            } else if let Some(split_tree) = self.push_tree_recursive(other, cx, arena) {
                return Self::from_child_trees(self, split_tree, cx);
            }
        }
        self
    }

    fn push_tree_recursive(
        &mut self,
        other: SumTree<T>,
        cx: &<T::Summary as Summary>::Context,
        arena: &mut Arena<T>,
    ) -> Option<SumTree<T>> {
        match Arc::make_mut(&mut self.0) {
            Node::Internal {
                height,
                child_summaries,
                child_trees,
                ..
            } => {
                <T::Summary as Summary>::add_summary(&mut self.1, other.summary(), cx);

                let height_delta = *height - other.0.height();
                let mut summaries_to_append = ArrayVec::<T::Summary, { 2 * TREE_BASE }>::new();
                let mut trees_to_append = ArrayVec::<SumTree<T>, { 2 * TREE_BASE }>::new();
                if height_delta == 0 {
                    summaries_to_append.extend(other.0.child_summaries().iter().cloned());
                    trees_to_append.extend(other.0.child_trees().iter().cloned());
                } else if height_delta == 1 && !other.0.is_underflowing() {
                    summaries_to_append.push(*other.summary());
                    trees_to_append.push(other)
                } else {
                    let tree_to_append = child_trees
                        .last_mut()
                        .unwrap()
                        .push_tree_recursive(other, cx, arena);
                    *child_summaries.last_mut().unwrap() =
                        *child_trees.last().unwrap().summary();

                    if let Some(split_tree) = tree_to_append {
                        summaries_to_append.push(*split_tree.summary());
                        trees_to_append.push(split_tree);
                    }
                }

                let child_count = child_trees.len() + trees_to_append.len();
                if child_count > 2 * TREE_BASE {
                    let left_summaries: ArrayVec<_, { 2 * TREE_BASE }>;
                    let right_summaries: ArrayVec<_, { 2 * TREE_BASE }>;
                    let left_trees;
                    let right_trees;

                    let midpoint = (child_count + child_count % 2) / 2;
                    {
                        let mut all_summaries = child_summaries
                            .iter()
                            .chain(summaries_to_append.iter())
                            .cloned();
                        left_summaries = all_summaries.by_ref().take(midpoint).collect();
                        right_summaries = all_summaries.collect();
                        let mut all_trees =
                            child_trees.iter().chain(trees_to_append.iter()).cloned();
                        left_trees = all_trees.by_ref().take(midpoint).collect();
                        right_trees = all_trees.collect();
                    }
                    self.1 = sum(left_summaries.iter(), cx);
                    *child_summaries = left_summaries;
                    *child_trees = left_trees;

                    Some(SumTree {
                        1: sum(right_summaries.iter(), cx),
                        0: Arc::new(Node::Internal {
                            height: *height,
                            child_summaries: right_summaries,
                            child_trees: right_trees,
                        }),
                    })
                } else {
                    child_summaries.extend(summaries_to_append);
                    child_trees.extend(trees_to_append);
                    None
                }
            }
            Node::Leaf {
                items,
                item_summaries,
            } => {
                let other_node = &*other.0;

                let child_count = items.len() + other_node.items().len();
                if child_count > 2 * TREE_BASE {
                    let left_items;
                    let right_items;
                    let left_summaries;
                    let right_summaries: ArrayVec<T::Summary, { 2 * TREE_BASE }>;

                    let midpoint = (child_count + child_count % 2) / 2;
                    {
                        let mut all_items = items.iter().chain(other_node.items().iter()).cloned();
                        left_items = all_items.by_ref().take(midpoint).collect();
                        right_items = all_items.collect();

                        let mut all_summaries = item_summaries
                            .iter()
                            .chain(other_node.child_summaries())
                            .cloned();
                        left_summaries = all_summaries.by_ref().take(midpoint).collect();
                        right_summaries = all_summaries.collect();
                    }
                    *items = left_items;
                    *item_summaries = left_summaries;
                    self.1 = sum(item_summaries.iter(), cx);

                    Some(SumTree {
                        1: sum(right_summaries.iter(), cx),
                        0: Arc::new(Node::Leaf {
                            items: right_items,
                            item_summaries: right_summaries,
                        }),
                    })
                } else {
                    <T::Summary as Summary>::add_summary(&mut self.1, other.summary(), cx);
                    items.extend(other_node.items().iter().cloned());
                    item_summaries.extend(other_node.child_summaries().iter().cloned());
                    None
                }
            }
        }
    }
}

impl<T: Item + PartialEq> PartialEq for SumTree<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: Item + Eq> Eq for SumTree<T> {}

struct NodeRef<'a, T: Item> {
    arena: &'a Arena<T>,
    node: NodeKey,
}

impl<T: Item> Deref for NodeRef<'_, T> {
    type Target = Node<T>;

    fn deref(&self) -> &Self::Target {
        self.arena.0.get(self.node).unwrap()
    }
}

struct NodeMut<'a, T: Item> {
    arena: &'a mut Arena<T>,
    node: NodeKey,
}

impl<T: Item> Deref for NodeMut<'_, T> {
    type Target = Node<T>;

    fn deref(&self) -> &Self::Target {
        self.arena.0.get(self.node).unwrap()
    }
}
impl<T: Item> DerefMut for NodeMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.arena.0.get_mut(self.node).unwrap()
    }
}

#[derive(Clone)]
enum Node<T: Item> {
    Internal {
        height: u8,
        child_summaries: ArrayVec<T::Summary, { 2 * TREE_BASE }>,
        child_trees: ArrayVec<SumTree<T>, { 2 * TREE_BASE }>,
    },
    Leaf {
        items: ArrayVec<T, { 2 * TREE_BASE }>,
        item_summaries: ArrayVec<T::Summary, { 2 * TREE_BASE }>,
    },
}

// impl<T> fmt::Debug for Node<T>
// where
//     T: Item + fmt::Debug,
//     T::Summary: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             Node::Internal {
//                 height,
//                 summary,
//                 child_summaries,
//                 child_trees,
//             } => f
//                 .debug_struct("Internal")
//                 .field("height", height)
//                 .field("summary", summary)
//                 .field("child_summaries", child_summaries)
//                 .field("child_trees", child_trees)
//                 .finish(),
//             Node::Leaf {
//                 summary,
//                 items,
//                 item_summaries,
//             } => f
//                 .debug_struct("Leaf")
//                 .field("summary", summary)
//                 .field("items", items)
//                 .field("item_summaries", item_summaries)
//                 .finish(),
//         }
//     }
// }

impl<T: Item> Node<T> {
    fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }

    fn height(&self) -> u8 {
        match self {
            Node::Internal { height, .. } => *height,
            Node::Leaf { .. } => 0,
        }
    }

    // fn summary(&self) -> &T::Summary {
    //     match self {
    //         Node::Internal { summary, .. } => summary,
    //         Node::Leaf { summary, .. } => summary,
    //     }
    // }

    fn child_summaries(&self) -> &[T::Summary] {
        match self {
            Node::Internal {
                child_summaries, ..
            } => child_summaries.as_slice(),
            Node::Leaf { item_summaries, .. } => item_summaries.as_slice(),
        }
    }

    fn child_trees(&self) -> &ArrayVec<SumTree<T>, { 2 * TREE_BASE }> {
        match self {
            Node::Internal { child_trees, .. } => child_trees,
            Node::Leaf { .. } => panic!("Leaf nodes have no child trees"),
        }
    }

    fn items(&self) -> &ArrayVec<T, { 2 * TREE_BASE }> {
        match self {
            Node::Leaf { items, .. } => items,
            Node::Internal { .. } => panic!("Internal nodes have no items"),
        }
    }

    fn is_underflowing(&self) -> bool {
        match self {
            Node::Internal { child_trees, .. } => child_trees.len() < TREE_BASE,
            Node::Leaf { items, .. } => items.len() < TREE_BASE,
        }
    }
}

fn sum<'a, T, I>(iter: I, cx: &T::Context) -> T
where
    T: 'a + Summary,
    I: Iterator<Item = &'a T>,
{
    let mut sum = T::zero(cx);
    for value in iter {
        sum.add_summary(value, cx);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distributions, prelude::*};
    use std::cmp;

    #[test]
    fn test_extend_and_push_tree() {
        let mut arena = Arena::default();
        let mut tree1 = SumTree::new(&(), &mut arena);
        tree1 = tree1.extend(0..20, &(), &mut arena);

        let mut tree2 = SumTree::new(&(), &mut arena);
        tree2 = tree2.extend(50..100, &(), &mut arena);

        tree1 = tree1.append(tree2, &(), &mut arena);
        assert_eq!(
            tree1.items(&(), &arena),
            (0..20).chain(50..100).collect::<Vec<u8>>()
        );
    }

    #[test]
    fn test_random() {
        let mut arena = Arena::default();

        let mut starting_seed = 0;
        if let Ok(value) = std::env::var("SEED") {
            starting_seed = value.parse().expect("invalid SEED variable");
        }
        let mut num_iterations = 100;
        if let Ok(value) = std::env::var("ITERATIONS") {
            num_iterations = value.parse().expect("invalid ITERATIONS variable");
        }
        let num_operations = std::env::var("OPERATIONS")
            .map_or(5, |o| o.parse().expect("invalid OPERATIONS variable"));

        for seed in starting_seed..(starting_seed + num_iterations) {
            let mut rng = StdRng::seed_from_u64(dbg!(seed));

            let rng = &mut rng;
            let mut tree = SumTree::<u8>::new(&(), &mut arena);
            let count = rng.gen_range(0..10);
            tree = tree.extend(
                rng.sample_iter(distributions::Standard).take(count),
                &(),
                &mut arena,
            );

            for _ in 0..num_operations {
                let splice_end = rng.gen_range(0..tree.extent::<Count>(&()).0 + 1);
                let splice_start = rng.gen_range(0..splice_end + 1);
                let count = rng.gen_range(0..10);
                let tree_end = tree.extent::<Count>(&());
                let new_items = rng
                    .sample_iter(distributions::Standard)
                    .take(count)
                    .collect::<Vec<u8>>();

                let mut reference_items = tree.items(&(), &arena);
                reference_items.splice(splice_start..splice_end, new_items.clone());

                tree = {
                    let mut cursor = tree.cursor::<Count>(&());
                    let mut new_tree = cursor.slice(&Count(splice_start), Bias::Right, &mut arena);
                    new_tree = new_tree.extend(new_items, &(), &mut arena);
                    cursor.seek(&Count(splice_end), Bias::Right, &mut arena);
                    new_tree.append(
                        cursor.slice(&tree_end, Bias::Right, &mut arena),
                        &(),
                        &mut arena,
                    )
                };

                assert_eq!(tree.items(&(), &arena), reference_items);
                // assert_eq!(
                //     tree.iter().collect::<Vec<_>>(),
                //     tree.cursor::<()>(&()).collect::<Vec<_>>()
                // );

                let mut before_start = false;
                let mut cursor = tree.cursor::<Count>(&());
                let start_pos = rng.gen_range(0..=reference_items.len());
                cursor.seek(&Count(start_pos), Bias::Right, &mut arena);
                let mut pos = rng.gen_range(start_pos..=reference_items.len());
                cursor.seek_forward(&Count(pos), Bias::Right, &mut arena);

                for i in 0..10 {
                    // assert_eq!(cursor.start().0, pos);

                    // if pos > 0 {
                    //     assert_eq!(cursor.prev_item().unwrap(), &reference_items[pos - 1]);
                    // } else {
                    //     assert_eq!(cursor.prev_item(), None);
                    // }

                    if pos < reference_items.len() && !before_start {
                        assert_eq!(cursor.item(&arena).unwrap(), &reference_items[pos]);
                    } else {
                        assert_eq!(cursor.item(&arena), None);
                    }

                    if before_start {
                        assert_eq!(cursor.next_item(&arena), reference_items.first());
                    } else if pos + 1 < reference_items.len() {
                        assert_eq!(cursor.next_item(&arena).unwrap(), &reference_items[pos + 1]);
                    } else {
                        assert_eq!(cursor.next_item(&arena), None);
                    }

                    if i < 5 {
                        cursor.next();
                        if pos < reference_items.len() {
                            pos += 1;
                            before_start = false;
                        }
                    } else {
                        // cursor.prev();
                        // if pos == 0 {
                        //     before_start = true;
                        // }
                        // pos = pos.saturating_sub(1);
                    }
                }
            }

            // for _ in 0..10 {
            //     let end = rng.gen_range(0..tree.extent::<Count>(&()).0 + 1);
            //     let start = rng.gen_range(0..end + 1);
            //     let start_bias = if rng.r#gen() { Bias::Left } else { Bias::Right };
            //     let end_bias = if rng.r#gen() { Bias::Left } else { Bias::Right };

            //     let mut cursor = tree.cursor::<Count>(&());
            //     cursor.seek(&Count(start), start_bias);
            //     let slice = cursor.slice(&Count(end), end_bias);

            //     cursor.seek(&Count(start), start_bias);
            //     let summary = cursor.summary::<_, Sum>(&Count(end), end_bias);

            //     assert_eq!(summary.0, slice.summary().sum);
            // }
        }
    }

    #[test]
    fn test_cursor() {
        let mut arena = Arena::default();

        // Empty tree
        let tree = SumTree::<u8>::new(&(), &mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&());
        assert_eq!(
            cursor
                .slice(&Count(0), Bias::Right, &mut arena)
                .items(&(), &arena),
            Vec::<u8>::new()
        );
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), None);
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 0);
        // cursor.prev();
        // assert_eq!(cursor.item(), None);
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 0);
        // cursor.next();
        // assert_eq!(cursor.item(), None);
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 0);

        // Single-element tree
        let tree = SumTree::<u8>::new(&(), &mut arena).extend(vec![1], &(), &mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&());
        assert_eq!(
            cursor
                .slice(&Count(0), Bias::Right, &mut arena)
                .items(&(), &arena),
            Vec::<u8>::new()
        );
        assert_eq!(cursor.item(&arena), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 0);

        cursor.next();
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 1);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 0);

        let mut cursor = tree.cursor::<IntegersSummary>(&());
        assert_eq!(
            cursor
                .slice(&Count(1), Bias::Right, &mut arena)
                .items(&(), &arena),
            [1]
        );
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 1);

        cursor.seek(&Count(0), Bias::Right, &mut arena);
        assert_eq!(
            cursor
                .slice(&tree.extent::<Count>(&()), Bias::Right, &mut arena)
                .items(&(), &arena),
            [1]
        );
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 1);

        // Multiple-element tree
        let tree = SumTree::new(&(), &mut arena).extend(vec![1, 2, 3, 4, 5, 6], &(), &mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&());

        assert_eq!(
            cursor
                .slice(&Count(2), Bias::Right, &mut arena)
                .items(&(), &arena),
            [1, 2]
        );
        assert_eq!(cursor.item(&arena), Some(&3));
        // assert_eq!(cursor.prev_item(), Some(&2));
        assert_eq!(cursor.next_item(&arena), Some(&4));
        // assert_eq!(cursor.start().sum, 3);

        cursor.next();
        assert_eq!(cursor.item(&arena), Some(&4));
        // assert_eq!(cursor.prev_item(), Some(&3));
        assert_eq!(cursor.next_item(&arena), Some(&5));
        // assert_eq!(cursor.start().sum, 6);

        cursor.next();
        assert_eq!(cursor.item(&arena), Some(&5));
        // assert_eq!(cursor.prev_item(), Some(&4));
        assert_eq!(cursor.next_item(&arena), Some(&6));
        // assert_eq!(cursor.start().sum, 10);

        cursor.next();
        assert_eq!(cursor.item(&arena), Some(&6));
        // assert_eq!(cursor.prev_item(), Some(&5));
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 15);

        cursor.next();
        cursor.next();
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&6));
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 21);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&6));
        // assert_eq!(cursor.prev_item(), Some(&5));
        // assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 15);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&5));
        // assert_eq!(cursor.prev_item(), Some(&4));
        // assert_eq!(cursor.next_item(&arena), Some(&6));
        // assert_eq!(cursor.start().sum, 10);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&4));
        // assert_eq!(cursor.prev_item(), Some(&3));
        // assert_eq!(cursor.next_item(&arena), Some(&5));
        // assert_eq!(cursor.start().sum, 6);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&3));
        // assert_eq!(cursor.prev_item(), Some(&2));
        // assert_eq!(cursor.next_item(&arena), Some(&4));
        // assert_eq!(cursor.start().sum, 3);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&2));
        // assert_eq!(cursor.prev_item(), Some(&1));
        // assert_eq!(cursor.next_item(&arena), Some(&3));
        // assert_eq!(cursor.start().sum, 1);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), Some(&2));
        // assert_eq!(cursor.start().sum, 0);

        // cursor.prev();
        // assert_eq!(cursor.item(), None);
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), Some(&1));
        // assert_eq!(cursor.start().sum, 0);

        // cursor.next();
        // assert_eq!(cursor.item(), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), Some(&2));
        // assert_eq!(cursor.start().sum, 0);

        let mut cursor = tree.cursor::<IntegersSummary>(&());
        assert_eq!(
            cursor
                .slice(&tree.extent::<Count>(&()), Bias::Right, &mut arena)
                .items(&(), &arena),
            tree.items(&(), &arena)
        );
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&6));
        // assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 21);

        cursor.seek(&Count(3), Bias::Right, &mut arena);
        assert_eq!(
            cursor
                .slice(&tree.extent::<Count>(&()), Bias::Right, &mut arena)
                .items(&(), &arena),
            [4, 5, 6]
        );
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&6));
        assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 21);

        // Seeking can bias left or right
        cursor.seek(&Count(1), Bias::Left, &mut arena);
        assert_eq!(cursor.item(&arena), Some(&1));
        cursor.seek(&Count(1), Bias::Right, &mut arena);
        assert_eq!(cursor.item(&arena), Some(&2));

        // Slicing without resetting starts from where the cursor is parked at.
        cursor.seek(&Count(1), Bias::Right, &mut arena);
        assert_eq!(
            cursor
                .slice(&Count(3), Bias::Right, &mut arena)
                .items(&(), &arena),
            vec![2, 3]
        );
        assert_eq!(
            cursor
                .slice(&Count(6), Bias::Left, &mut arena)
                .items(&(), &arena),
            vec![4, 5]
        );
        assert_eq!(
            cursor
                .slice(&Count(6), Bias::Right, &mut arena)
                .items(&(), &arena),
            vec![6]
        );
    }

    #[test]
    fn test_from_iter() {
        let mut arena = Arena::default();
        assert_eq!(
            SumTree::new(&(), &mut arena)
                .extend(0..100, &(), &mut arena)
                .items(&(), &arena),
            (0..100).collect::<Vec<_>>()
        );

        // Ensure `from_iter` works correctly when the given iterator restarts
        // after calling `next` if `None` was already returned.
        let mut ix = 0;
        let iterator = std::iter::from_fn(|| {
            ix = (ix + 1) % 2;
            if ix == 1 { Some(1) } else { None }
        });
        assert_eq!(
            SumTree::new(&(), &mut arena)
                .extend(iterator, &(), &mut arena)
                .items(&(), &arena),
            vec![1]
        );
    }

    #[derive(Copy, Clone, Default, Debug)]
    pub struct IntegersSummary {
        count: usize,
        sum: usize,
        contains_even: bool,
        max: u8,
    }

    #[derive(Copy, Ord, PartialOrd, Default, Eq, PartialEq, Clone, Debug)]
    struct Count(usize);

    #[derive(Copy, Ord, PartialOrd, Default, Eq, PartialEq, Clone, Debug)]
    struct Sum(usize);

    impl Item for u8 {
        type Summary = IntegersSummary;

        fn summary(&self, _cx: &()) -> Self::Summary {
            IntegersSummary {
                count: 1,
                sum: *self as usize,
                contains_even: (*self & 1) == 0,
                max: *self,
            }
        }
    }

    impl Summary for IntegersSummary {
        type Context = ();

        fn zero(_cx: &()) -> Self {
            Default::default()
        }

        fn add_summary(&mut self, other: &Self, _: &()) {
            self.count += other.count;
            self.sum += other.sum;
            self.contains_even |= other.contains_even;
            self.max = cmp::max(self.max, other.max);
        }
    }

    impl Dimension<'_, IntegersSummary> for u8 {
        fn zero(_cx: &()) -> Self {
            Default::default()
        }

        fn add_summary(&mut self, summary: &IntegersSummary, _: &()) {
            *self = summary.max;
        }
    }

    impl Dimension<'_, IntegersSummary> for Count {
        fn zero(_cx: &()) -> Self {
            Default::default()
        }

        fn add_summary(&mut self, summary: &IntegersSummary, _: &()) {
            self.0 += summary.count;
        }
    }

    impl SeekTarget<'_, IntegersSummary, IntegersSummary> for Count {
        fn cmp(&self, cursor_location: &IntegersSummary, _: &()) -> Ordering {
            std::cmp::Ord::cmp(&self.0, &cursor_location.count)
        }
    }

    impl Dimension<'_, IntegersSummary> for Sum {
        fn zero(_cx: &()) -> Self {
            Default::default()
        }

        fn add_summary(&mut self, summary: &IntegersSummary, _: &()) {
            self.0 += summary.sum;
        }
    }
}
