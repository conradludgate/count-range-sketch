mod cursor;

use arrayvec::ArrayVec;
pub use cursor::Cursor;
use slotmap::SlotMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::{cmp::Ordering, fmt, iter::FromIterator};

use crate::sumtree::cursor::IntoCursor;

#[cfg(test)]
pub const TREE_BASE: usize = 2;
#[cfg(not(test))]
pub const TREE_BASE: usize = 6;

slotmap::new_key_type! {
    struct NodeKey;
}

pub struct Arena<T: Item>(SlotMap<NodeKey, Node<T>>);

impl<T: Item> Arena<T> {
    fn alloc(&mut self, summary: T::Summary, node: Node<T>) -> SumTree<T> {
        SumTree {
            node: Box::new(node),
            summary,
        }
    }

    fn remove(&mut self, sumtree: SumTree<T>) -> Node<T> {
        *sumtree.node
    }

    pub fn drop(&mut self, sumtree: SumTree<T>) {
        match self.remove(sumtree) {
            Node::Internal { child_trees, .. } => {
                child_trees.into_iter().for_each(|tree| self.drop(tree));
            }
            Node::Leaf { .. } => {}
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T: Item> Default for Arena<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// An item that can be stored in a [`SumTree`]
///
/// Must be summarized by a type that implements [`Summary`]
pub trait Item {
    type Summary: Summary;

    fn summary(&self, cx: &<Self::Summary as Summary>::Context) -> Self::Summary;
}

/// A type that describes the Sum of all [`Item`]s in a subtree of the [`SumTree`]
///
/// Each Summary type can have multiple [`Dimension`]s that it measures,
/// which can be used to navigate the tree
pub trait Summary: Copy + std::fmt::Debug {
    type Context;

    fn zero(cx: &Self::Context) -> Self;
    fn add_summary(&mut self, summary: Self, cx: &Self::Context);
}

/// Each [`Summary`] type can have more than one [`Dimension`] type that it measures.
///
/// You can use dimensions to seek to a specific location in the [`SumTree`]
///
/// # Example:
/// Zed's rope has a `TextSummary` type that summarizes lines, characters, and bytes.
/// Each of these are different dimensions we may want to seek to
pub trait Dimension<S: Summary>: Copy {
    fn zero(cx: &S::Context) -> Self;

    fn add_summary(&mut self, summary: S, cx: &S::Context);
}

impl<T: Summary> Dimension<T> for T {
    fn zero(cx: &T::Context) -> Self {
        Summary::zero(cx)
    }

    fn add_summary(&mut self, summary: T, cx: &T::Context) {
        Summary::add_summary(self, summary, cx);
    }
}

pub trait SeekTarget<S: Summary, D: Dimension<S>> {
    fn cmp(&self, cursor_location: &D, cx: &S::Context) -> Ordering;
}

impl<S: Summary, D: Dimension<S> + Ord> SeekTarget<S, D> for D {
    fn cmp(&self, cursor_location: &Self, _: &S::Context) -> Ordering {
        Ord::cmp(self, cursor_location)
    }
}

impl<T: Summary> Dimension<T> for () {
    fn zero(_: &T::Context) -> Self {}

    fn add_summary(&mut self, _: T, _: &T::Context) {}
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
pub struct SumTree<T: Item> {
    node: Box<Node<T>>,
    pub summary: T::Summary,
}

impl<T> fmt::Debug for SumTree<T>
where
    T: fmt::Debug + Item,
    T::Summary: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SumTree")
            .field("node", &self.node)
            .field("summary", &self.summary)
            .finish()
    }
}

impl<T: Item> SumTree<T> {
    pub fn new(cx: &<T::Summary as Summary>::Context, arena: &mut Arena<T>) -> Self {
        SumTree::from_summary(<T::Summary as Summary>::zero(cx), arena)
    }

    /// Useful in cases where the item type has a non-trivial context type, but the zero value of the summary type doesn't depend on that context.
    pub fn from_summary(summary: T::Summary, arena: &mut Arena<T>) -> Self {
        arena.alloc(
            summary,
            Node::Leaf {
                items: ArrayVec::new(),
            },
        )
    }

    pub fn from_item(item: T, cx: &<T::Summary as Summary>::Context, arena: &mut Arena<T>) -> Self {
        let summary = item.summary(cx);
        arena.alloc(
            summary,
            Node::Leaf {
                items: ArrayVec::from_iter(Some(item)),
            },
        )
    }

    pub fn cursor<'a, S>(&'a self, cx: &'a <T::Summary as Summary>::Context) -> Cursor<'a, T, S>
    where
        S: Dimension<T::Summary>,
    {
        Cursor::new(self, cx)
    }

    pub fn into_cursor<'a, S>(
        self,
        cx: &'a <T::Summary as Summary>::Context,
    ) -> IntoCursor<'a, T, S>
    where
        S: Dimension<T::Summary>,
    {
        IntoCursor::new(self, cx)
    }

    #[allow(dead_code)]
    pub fn first(&self) -> Option<&T> {
        self.leftmost_leaf().node.items().first()
    }

    // pub fn last(&self) -> Option<&T> {
    //     self.rightmost_leaf().0.items().last()
    // }

    #[cfg(test)]
    pub fn extent<D: Dimension<T::Summary>>(&self, cx: &<T::Summary as Summary>::Context) -> D {
        let mut extent = D::zero(cx);
        extent.add_summary(self.summary, cx);
        extent
    }

    pub fn is_empty(&self) -> bool {
        match self.node.as_ref() {
            Node::Internal { .. } => false,
            Node::Leaf { items, .. } => items.is_empty(),
        }
    }

    fn from_child_trees(
        left: SumTree<T>,
        right: SumTree<T>,
        cx: &<T::Summary as Summary>::Context,
        arena: &mut Arena<T>,
    ) -> Self {
        let height = left.node.height() + 1;
        let summary = sum([left.summary, right.summary], cx);

        let mut child_trees = ArrayVec::new();
        child_trees.push(left);
        child_trees.push(right);

        arena.alloc(
            summary,
            Node::Internal {
                height,
                child_trees,
            },
        )
    }

    fn leftmost_leaf(&self) -> &Self {
        match *self.node {
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

    pub fn items(&self, cx: &<T::Summary as Summary>::Context, arena: &Arena<T>) -> Vec<T>
    where
        T: Clone,
    {
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
        self.append(Self::from_item(item, cx, arena), cx, arena)
    }

    pub fn append(
        mut self,
        other: Self,
        cx: &<T::Summary as Summary>::Context,
        arena: &mut Arena<T>,
    ) -> Self {
        if self.is_empty() {
            return other;
        } else if !other.node.is_leaf() || !other.node.items().is_empty() {
            if self.node.height() < other.node.height() {
                for tree in other.node.into_child_trees() {
                    self = self.append(tree, cx, arena);
                }
            } else if let Some(split_tree) = self.push_tree_recursive(other, cx, arena) {
                return Self::from_child_trees(self, split_tree, cx, arena);
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
        match &mut *self.node {
            Node::Internal {
                height,
                child_trees,
            } => {
                <T::Summary as Summary>::add_summary(&mut self.summary, other.summary, cx);

                let height_delta = *height - other.node.height();
                let mut trees_to_append = ArrayVec::<SumTree<T>, { 2 * TREE_BASE }>::new();
                if height_delta == 0 {
                    let Node::Internal { child_trees, .. } = *other.node else {
                        unreachable!()
                    };

                    trees_to_append.extend(child_trees);
                } else if height_delta == 1 && !other.node.is_underflowing() {
                    trees_to_append.push(other)
                } else {
                    let tree_to_append = child_trees
                        .last_mut()
                        .unwrap()
                        .push_tree_recursive(other, cx, arena);

                    if let Some(split_tree) = tree_to_append {
                        trees_to_append.push(split_tree);
                    }
                }

                let child_count = child_trees.len() + trees_to_append.len();
                if child_count > 2 * TREE_BASE {
                    let midpoint = (child_count + child_count % 2) / 2;

                    let mut all_trees = child_trees.drain(..).chain(trees_to_append.drain(..));
                    let left_trees: ArrayVec<_, { 2 * TREE_BASE }> =
                        all_trees.by_ref().take(midpoint).collect();
                    let right_trees: ArrayVec<_, { 2 * TREE_BASE }> = all_trees.collect();

                    self.summary = sum(left_trees.iter().map(|tree| tree.summary), cx);
                    *child_trees = left_trees;

                    Some(arena.alloc(
                        sum(right_trees.iter().map(|tree| tree.summary), cx),
                        Node::Internal {
                            height: *height,
                            child_trees: right_trees,
                        },
                    ))
                } else {
                    child_trees.extend(trees_to_append);
                    None
                }
            }
            Node::Leaf { items } => {
                let Node::Leaf { items: other_items } = *other.node else {
                    unreachable!()
                };

                let child_count = items.len() + other_items.len();
                if child_count > 2 * TREE_BASE {
                    let midpoint = (child_count + child_count % 2) / 2;

                    let mut all_items = items.drain(..).chain(other_items);
                    let left_items: ArrayVec<T, { 2 * TREE_BASE }> =
                        all_items.by_ref().take(midpoint).collect();
                    let right_items: ArrayVec<T, { 2 * TREE_BASE }> = all_items.collect();

                    self.summary = sum(left_items.iter().map(|item| item.summary(cx)), cx);
                    *items = left_items;

                    Some(arena.alloc(
                        sum(right_items.iter().map(|item| item.summary(cx)), cx),
                        Node::Leaf { items: right_items },
                    ))
                } else {
                    <T::Summary as Summary>::add_summary(&mut self.summary, other.summary, cx);
                    items.extend(other_items);
                    None
                }
            }
        }
    }
}

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

enum Node<T: Item> {
    Internal {
        height: u8,
        child_trees: ArrayVec<SumTree<T>, { 2 * TREE_BASE }>,
    },
    Leaf {
        items: ArrayVec<T, { 2 * TREE_BASE }>,
    },
}

impl<T> fmt::Debug for Node<T>
where
    T: Item + fmt::Debug,
    T::Summary: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Internal {
                height,
                child_trees,
            } => f
                .debug_struct("Internal")
                .field("height", height)
                .field("child_trees", child_trees)
                .finish(),
            Node::Leaf { items } => f.debug_struct("Leaf").field("items", items).finish(),
        }
    }
}

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

    // fn child_summaries(&self) -> &[T::Summary] {
    //     match self {
    //         Node::Internal {
    //             child_summaries, ..
    //         } => child_summaries.as_slice(),
    //         Node::Leaf { item_summaries, .. } => item_summaries.as_slice(),
    //     }
    // }

    fn into_child_trees(self) -> ArrayVec<SumTree<T>, { 2 * TREE_BASE }> {
        match self {
            Node::Internal { child_trees, .. } => child_trees,
            Node::Leaf { .. } => panic!("Leaf nodes have no child trees"),
        }
    }

    #[cfg(test)]
    fn child_trees(&self) -> &ArrayVec<SumTree<T>, { 2 * TREE_BASE }> {
        match self {
            Node::Internal { child_trees, .. } => child_trees,
            Node::Leaf { .. } => panic!("Leaf nodes have no child trees"),
        }
    }

    // fn into_items(self) -> ArrayVec<T, { 2 * TREE_BASE }> {
    //     match self {
    //         Node::Leaf { items, .. } => items,
    //         Node::Internal { .. } => panic!("Internal nodes have no items"),
    //     }
    // }

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
    I: IntoIterator<Item = T>,
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
                    let mut cursor = tree.into_cursor::<Count>(&());
                    let mut new_tree = cursor.slice(&Count(splice_start), Bias::Right, &mut arena);
                    new_tree = new_tree.extend(new_items, &(), &mut arena);
                    let _ = cursor.slice(&Count(splice_end), Bias::Right, &mut arena);
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
                cursor.seek_forward(&Count(start_pos), Bias::Right, &arena);
                let mut pos = rng.gen_range(start_pos..=reference_items.len());
                cursor.seek_forward(&Count(pos), Bias::Right, &arena);

                for i in 0..10 {
                    assert_eq!(cursor.start().0, pos);

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
            //     cursor.seek(&Count(start), start_bias, &arena);
            //     let slice = cursor.slice(&Count(end), end_bias);

            //     cursor.seek(&Count(start), start_bias, &arena);
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
        cursor.seek_forward(&Count(0), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), None);
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 0);
        // cursor.prev();
        // assert_eq!(cursor.item(), None);
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 0);
        // cursor.next();
        // assert_eq!(cursor.item(), None);
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 0);

        // Single-element tree
        let tree = SumTree::<u8>::new(&(), &mut arena).extend(vec![1], &(), &mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&());
        cursor.seek_forward(&Count(0), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 0);

        cursor.next();
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 1);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 0);

        let mut cursor = tree.cursor::<IntegersSummary>(&());
        cursor.seek_forward(&tree.extent::<Count>(&()), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 1);

        cursor.seek(&Count(0), Bias::Right, &arena);
        cursor.seek_forward(&tree.extent::<Count>(&()), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 1);

        // Multiple-element tree
        let tree = SumTree::new(&(), &mut arena).extend(vec![1, 2, 3, 4, 5, 6], &(), &mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&());

        cursor.seek_forward(&Count(2), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), Some(&3));
        // assert_eq!(cursor.prev_item(), Some(&2));
        assert_eq!(cursor.next_item(&arena), Some(&4));
        assert_eq!(cursor.start().sum, 3);

        cursor.next();
        assert_eq!(cursor.item(&arena), Some(&4));
        // assert_eq!(cursor.prev_item(), Some(&3));
        assert_eq!(cursor.next_item(&arena), Some(&5));
        assert_eq!(cursor.start().sum, 6);

        cursor.next();
        assert_eq!(cursor.item(&arena), Some(&5));
        // assert_eq!(cursor.prev_item(), Some(&4));
        assert_eq!(cursor.next_item(&arena), Some(&6));
        assert_eq!(cursor.start().sum, 10);

        cursor.next();
        assert_eq!(cursor.item(&arena), Some(&6));
        // assert_eq!(cursor.prev_item(), Some(&5));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 15);

        cursor.next();
        cursor.next();
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&6));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 21);

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
        cursor.seek_forward(&tree.extent::<Count>(&()), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&6));
        // assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 21);

        cursor.seek(&Count(3), Bias::Right, &arena);
        cursor.seek_forward(&tree.extent::<Count>(&()), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&6));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 21);

        // Seeking can bias left or right
        cursor.seek(&Count(1), Bias::Left, &arena);
        assert_eq!(cursor.item(&arena), Some(&1));
        cursor.seek(&Count(1), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), Some(&2));

        // Slicing without resetting starts from where the cursor is parked at.
        cursor.seek(&Count(1), Bias::Right, &arena);
        cursor.seek_forward(&Count(3), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), Some(&4));
        cursor.seek_forward(&Count(6), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        cursor.seek_forward(&Count(6), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
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

        fn add_summary(&mut self, other: Self, _: &()) {
            self.count += other.count;
            self.sum += other.sum;
            self.contains_even |= other.contains_even;
            self.max = cmp::max(self.max, other.max);
        }
    }

    impl Dimension<IntegersSummary> for u8 {
        fn zero(_cx: &()) -> Self {
            Default::default()
        }

        fn add_summary(&mut self, summary: IntegersSummary, _: &()) {
            *self = summary.max;
        }
    }

    impl Dimension<IntegersSummary> for Count {
        fn zero(_cx: &()) -> Self {
            Default::default()
        }

        fn add_summary(&mut self, summary: IntegersSummary, _: &()) {
            self.0 += summary.count;
        }
    }

    impl SeekTarget<IntegersSummary, IntegersSummary> for Count {
        fn cmp(&self, cursor_location: &IntegersSummary, _: &()) -> Ordering {
            std::cmp::Ord::cmp(&self.0, &cursor_location.count)
        }
    }

    impl Dimension<IntegersSummary> for Sum {
        fn zero(_cx: &()) -> Self {
            Default::default()
        }

        fn add_summary(&mut self, summary: IntegersSummary, _: &()) {
            self.0 += summary.sum;
        }
    }
}
