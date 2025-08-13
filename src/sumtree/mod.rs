mod cursor;

use arrayvec::ArrayVec;
pub use cursor::Cursor;
use equivalent::{Comparable, Equivalent};
use get_size2::GetSize;
use slotmap::SlotMap;
use std::ops::{Add, Deref, DerefMut};
use std::{fmt, iter::FromIterator};

use crate::sumtree::cursor::IntoCursor;

#[cfg(test)]
pub const TREE_BASE: usize = 2;
#[cfg(not(test))]
pub const TREE_BASE: usize = 6;

slotmap::new_key_type! {
    struct NodeKey;
}

impl GetSize for NodeKey {}

pub struct Arena<T: Item>(SlotMap<NodeKey, Node<T>>);

impl<T: Item + GetSize> GetSize for Arena<T>
where
    T::Summary: GetSize,
{
    fn get_heap_size(&self) -> usize {
        let mut size = self.0.capacity() * size_of::<T>();
        for node in self.0.iter() {
            size += node.1.get_heap_size();
        }
        size
    }
}

impl<T: Item> Arena<T> {
    fn alloc(&mut self, summary: T::Summary, node: Node<T>) -> InnerSumTree<T> {
        InnerSumTree {
            summary,
            node: self.0.insert(node),
        }
    }

    fn remove(&mut self, sumtree: InnerSumTree<T>) -> Node<T> {
        self.0.remove(sumtree.node).unwrap()
    }

    pub fn reset(&mut self, tree: &mut SumTree<T>)
    where
        T::Summary: Min,
    {
        if cfg!(debug_assertions) {
            tree.inner.summary = Min::MIN;

            let node = std::mem::replace(
                tree.inner.get_mut(self).into(),
                Node::Leaf {
                    items: ArrayVec::new(),
                },
            );
            if let Node::Internal { child_trees, .. } = node {
                child_trees
                    .into_iter()
                    .for_each(|tree| self.drop_inner(tree));
            }
        } else {
            self.clear();
            *tree = SumTree::new(self);
        }
    }

    pub fn drop(&mut self, sumtree: SumTree<T>) {
        self.drop_inner(sumtree.inner);
    }

    fn drop_inner(&mut self, sumtree: InnerSumTree<T>) {
        match self.remove(sumtree) {
            Node::Internal { child_trees, .. } => {
                child_trees
                    .into_iter()
                    .for_each(|tree| self.drop_inner(tree));
            }
            Node::Leaf { .. } => {}
        }
    }

    #[cfg(test)]
    pub fn assert_reachability(&self, tree: &SumTree<T>)
    where
        T: std::fmt::Debug,
        T::Summary: Min + std::fmt::Debug + Copy + Add<Output = T::Summary>,
    {
        if cfg!(debug_assertions) {
            let mut tracker = slotmap::SecondaryMap::with_capacity(self.0.len());
            for (k, _) in self.0.iter() {
                tracker.insert(k, ());
            }
            self.walk_reachability(&tree.inner, &mut tracker);
            let nodes: Vec<NodeRef<'_, T>> = tracker
                .into_iter()
                .map(|(node, ())| NodeRef {
                    node,
                    arena: self,
                    summary: self.0.get(node).unwrap().summary(),
                })
                .collect();
            assert!(nodes.is_empty(), "unreachable nodes found: {nodes:?}");
        }
    }

    #[cfg(test)]
    fn walk_reachability(
        &self,
        tree: &InnerSumTree<T>,
        tracker: &mut slotmap::SecondaryMap<NodeKey, ()>,
    ) where
        T::Summary: Copy,
    {
        tracker.remove(tree.node);
        match tree.get(self).into() {
            Node::Internal { child_trees, .. } => {
                child_trees
                    .into_iter()
                    .for_each(|tree| self.walk_reachability(tree, tracker));
            }
            Node::Leaf { .. } => {}
        }
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
    type Summary;

    fn summary(&self) -> Self::Summary;
}

pub trait Min {
    const MIN: Self;
}

#[derive(Clone, Copy)]
pub struct Empty;

impl Min for Empty {
    const MIN: Self = Empty;
}

// impl Summary for Empty {}

// impl<T: Summary> Dimension<T> for Empty {}

impl<T> Add<T> for Empty {
    type Output = Self;

    fn add(self, _: T) -> Self::Output {
        self
    }
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
    inner: InnerSumTree<T>,
    height: u8,
}

struct InnerSumTree<T: Item> {
    summary: T::Summary,
    node: NodeKey,
}

impl<T: Item + GetSize> GetSize for SumTree<T>
where
    T::Summary: GetSize,
{
    fn get_heap_size(&self) -> usize {
        self.inner.get_heap_size()
    }
}

impl<T: Item + GetSize> GetSize for InnerSumTree<T>
where
    T::Summary: GetSize,
{
    fn get_heap_size(&self) -> usize {
        self.summary.get_heap_size() + self.node.get_size()
    }
}

impl<T> SumTree<T>
where
    T: fmt::Debug + Item,
    T::Summary: fmt::Debug + Copy,
{
    pub fn fmt(&self, f: &mut fmt::Formatter, arena: &Arena<T>) -> fmt::Result {
        use fmt::Debug;
        self.inner.get(arena).fmt(f)
    }
}

impl<T: Item> SumTree<T> {
    pub fn new(arena: &mut Arena<T>) -> Self
    where
        T::Summary: Min,
    {
        SumTree::from_summary(Min::MIN, arena)
    }

    /// Useful in cases where the item type has a non-trivial context type, but the zero value of the summary type doesn't depend on that context.
    pub fn from_summary(summary: T::Summary, arena: &mut Arena<T>) -> Self {
        Self {
            inner: arena.alloc(
                summary,
                Node::Leaf {
                    items: ArrayVec::new(),
                },
            ),
            height: 0,
        }
    }

    pub fn from_item(item: T, arena: &mut Arena<T>) -> Self {
        let summary = item.summary();
        Self {
            inner: arena.alloc(
                summary,
                Node::Leaf {
                    items: ArrayVec::from_iter(Some(item)),
                },
            ),
            height: 0,
        }
    }

    pub fn summary(&self) -> &T::Summary {
        &self.inner.summary
    }

    pub fn cursor<'a, S>(&'a self, arena: &Arena<T>) -> Cursor<'a, T, S>
    where
        T::Summary: Copy,
        S: Add<T::Summary, Output = S> + Min + Copy,
    {
        Cursor::new(self, arena)
    }

    pub fn into_cursor<S>(self, arena: &mut Arena<T>) -> IntoCursor<T, S>
    where
        T::Summary: Min + Copy + Add<Output = T::Summary>,
        S: Add<T::Summary, Output = S> + Min + Copy,
    {
        IntoCursor::new(self, arena)
    }

    pub fn first<'a>(&'a self, arena: &'a Arena<T>) -> Option<&'a T>
    where
        T::Summary: Copy,
    {
        self.inner.leftmost_leaf(arena).into().items().first()
    }

    // pub fn last(&self) -> Option<&T> {
    //     self.rightmost_leaf().0.items().last()
    // }

    #[cfg(test)]
    pub fn extent<D: Add<T::Summary, Output = D> + Min>(&self) -> D
    where
        T::Summary: Copy,
    {
        D::MIN + self.inner.summary
    }

    pub fn is_empty(&self, arena: &Arena<T>) -> bool
    where
        T::Summary: Copy,
    {
        match self.inner.get(arena).into() {
            Node::Internal { .. } => false,
            Node::Leaf { items, .. } => items.is_empty(),
        }
    }

    fn from_child_trees(left: SumTree<T>, right: InnerSumTree<T>, arena: &mut Arena<T>) -> Self
    where
        T::Summary: Min + Copy + Add<Output = T::Summary>,
    {
        let Self {
            inner: left,
            height,
        } = left;

        let summary = sum([left.summary, right.summary]);

        SumTree {
            inner: arena.alloc(
                summary,
                Node::Internal {
                    child_trees: ArrayVec::from_iter([left, right]),
                },
            ),
            height: height + 1,
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

    pub fn items(&self, arena: &Arena<T>) -> Vec<T>
    where
        T: Clone,
        T::Summary: Copy,
    {
        let mut items = Vec::new();
        let mut cursor = self.cursor::<Empty>(arena);
        cursor.next(arena);
        while let Some(item) = cursor.item(arena) {
            items.push(item.clone());
            cursor.next(arena);
        }
        items
    }

    #[cfg(test)]
    pub fn extend<I>(self, iter: I, arena: &mut Arena<T>) -> Self
    where
        I: IntoIterator<Item = T>,
        T::Summary: Min + Copy + Add<Output = T::Summary>,
    {
        let tree = iter
            .into_iter()
            .fold(SumTree::new(arena), |tree, i| tree.push(i, arena));
        self.append(tree, arena)
    }

    pub fn push(mut self, item: T, arena: &mut Arena<T>) -> Self
    where
        T::Summary: Min + Copy + Add<Output = T::Summary>,
    {
        if self.is_empty(arena) {
            arena.remove(self.inner);
            return Self::from_item(item, arena);
        }

        if let Some(split_tree) = self.inner.push_item_recursive(item, arena) {
            Self::from_child_trees(self, split_tree, arena)
        } else {
            self
        }
    }

    pub fn append(mut self, other: Self, arena: &mut Arena<T>) -> Self
    where
        T::Summary: Min + Copy + Add<Output = T::Summary>,
    {
        if self.is_empty(arena) {
            arena.remove(self.inner);
            return other;
        }

        if other.is_empty(arena) {
            arena.remove(other.inner);
            return self;
        }

        if self.height < other.height {
            let children = arena.remove(other.inner).into_child_trees();
            let height = other.height - 1;
            children
                .into_iter()
                .fold(self, |l, inner| l.append(SumTree { inner, height }, arena))
        } else if let Some(split_tree) = self.inner.push_tree_recursive(self.height, other, arena) {
            Self::from_child_trees(self, split_tree, arena)
        } else {
            self
        }
    }
}

impl<T: Item> InnerSumTree<T> {
    fn get<'a>(&self, arena: &'a Arena<T>) -> NodeRef<'a, T>
    where
        T::Summary: Copy,
    {
        NodeRef {
            arena,
            node: self.node,
            summary: self.summary,
        }
    }

    fn get_mut<'a>(&mut self, arena: &'a mut Arena<T>) -> NodeMut<'a, T> {
        NodeMut {
            arena,
            node: self.node,
        }
    }

    fn leftmost_leaf<'a>(&self, arena: &'a Arena<T>) -> NodeRef<'a, T>
    where
        T::Summary: Copy,
    {
        let this = self.get(arena);
        match &*this {
            Node::Leaf { .. } => this,
            Node::Internal { child_trees, .. } => child_trees.first().unwrap().leftmost_leaf(arena),
        }
    }

    fn push_tree_recursive(
        &mut self,
        height: u8,
        other: SumTree<T>,
        arena: &mut Arena<T>,
    ) -> Option<InnerSumTree<T>>
    where
        T::Summary: Min + Copy + Add<Output = T::Summary>,
    {
        let SumTree {
            inner: other,
            height: other_height,
        } = other;

        let summary = self.summary + other.summary;

        match arena.0.remove(self.node).unwrap() {
            Node::Internal { mut child_trees } => {
                let height_delta = height - other_height;
                let mut trees_to_append = ArrayVec::<InnerSumTree<T>, { 2 * TREE_BASE }>::new();
                if height_delta == 0 {
                    let Node::Internal { child_trees, .. } = arena.remove(other) else {
                        unreachable!()
                    };

                    trees_to_append.extend(child_trees);
                } else if height_delta == 1 && !other.get(arena).is_underflowing() {
                    trees_to_append.push(other)
                } else {
                    let tree_to_append = child_trees.last_mut().unwrap().push_tree_recursive(
                        height - 1,
                        SumTree {
                            inner: other,
                            height: other_height,
                        },
                        arena,
                    );

                    if let Some(split_tree) = tree_to_append {
                        trees_to_append.push(split_tree);
                    }
                }

                let child_count = child_trees.len() + trees_to_append.len();
                if child_count > 2 * TREE_BASE {
                    let midpoint = (child_count + child_count % 2) / 2;

                    let mut all_trees = child_trees.into_iter().chain(trees_to_append);
                    let left_trees: ArrayVec<_, { 2 * TREE_BASE }> =
                        all_trees.by_ref().take(midpoint).collect();
                    let right_trees: ArrayVec<_, { 2 * TREE_BASE }> = all_trees.collect();

                    *self = arena.alloc(
                        sum(left_trees.iter().map(|tree| tree.summary)),
                        Node::Internal {
                            child_trees: left_trees,
                        },
                    );
                    Some(arena.alloc(
                        sum(right_trees.iter().map(|tree| tree.summary)),
                        Node::Internal {
                            child_trees: right_trees,
                        },
                    ))
                } else {
                    child_trees.extend(trees_to_append);

                    *self = arena.alloc(summary, Node::Internal { child_trees });
                    None
                }
            }
            Node::Leaf { mut items } => {
                let Node::Leaf { items: other_items } = arena.remove(other) else {
                    unreachable!()
                };

                let child_count = items.len() + other_items.len();
                if child_count > 2 * TREE_BASE {
                    let midpoint = (child_count + child_count % 2) / 2;

                    let mut all_items = items.into_iter().chain(other_items);
                    let left_items: ArrayVec<T, { 2 * TREE_BASE }> =
                        all_items.by_ref().take(midpoint).collect();
                    let right_items: ArrayVec<T, { 2 * TREE_BASE }> = all_items.collect();

                    *self = arena.alloc(
                        sum(left_items.iter().map(|item| item.summary())),
                        Node::Leaf { items: left_items },
                    );

                    Some(arena.alloc(
                        sum(right_items.iter().map(|item| item.summary())),
                        Node::Leaf { items: right_items },
                    ))
                } else {
                    items.extend(other_items);

                    *self = arena.alloc(summary, Node::Leaf { items });

                    None
                }
            }
        }
    }

    fn push_item_recursive(&mut self, item: T, arena: &mut Arena<T>) -> Option<InnerSumTree<T>>
    where
        T::Summary: Min + Copy + Add<Output = T::Summary>,
    {
        let summary = self.summary + item.summary();

        match arena.0.remove(self.node).unwrap() {
            Node::Internal { mut child_trees } => {
                let tree_to_append = child_trees
                    .last_mut()
                    .unwrap()
                    .push_item_recursive(item, arena);

                let child_count = child_trees.len() + tree_to_append.is_some() as usize;
                if child_count > 2 * TREE_BASE {
                    let midpoint = (child_count + child_count % 2) / 2;

                    let right_trees: ArrayVec<_, { 2 * TREE_BASE }> = child_trees
                        .drain(midpoint..)
                        .chain(tree_to_append)
                        .collect();

                    *self = arena.alloc(
                        sum(child_trees.iter().map(|tree| tree.summary)),
                        Node::Internal { child_trees },
                    );
                    Some(arena.alloc(
                        sum(right_trees.iter().map(|tree| tree.summary)),
                        Node::Internal {
                            child_trees: right_trees,
                        },
                    ))
                } else {
                    child_trees.extend(tree_to_append);

                    *self = arena.alloc(summary, Node::Internal { child_trees });
                    None
                }
            }
            Node::Leaf { mut items } => {
                let child_count = items.len() + 1;
                if child_count > 2 * TREE_BASE {
                    let midpoint = (child_count + child_count % 2) / 2;

                    let right_items: ArrayVec<T, { 2 * TREE_BASE }> = items
                        .drain(midpoint..)
                        .chain(std::iter::once(item))
                        .collect();

                    *self = arena.alloc(
                        sum(items.iter().map(|item| item.summary())),
                        Node::Leaf { items },
                    );

                    Some(arena.alloc(
                        sum(right_items.iter().map(|item| item.summary())),
                        Node::Leaf { items: right_items },
                    ))
                } else {
                    items.push(item);

                    *self = arena.alloc(summary, Node::Leaf { items });

                    None
                }
            }
        }
    }
}

struct NodeRef<'a, T: Item> {
    arena: &'a Arena<T>,
    node: NodeKey,
    summary: T::Summary,
}

impl<'a, T: Item> NodeRef<'a, T> {
    fn into(self) -> &'a Node<T> {
        self.arena.0.get(self.node).unwrap()
    }
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

impl<'a, T: Item> NodeMut<'a, T> {
    fn into(self) -> &'a mut Node<T> {
        self.arena.0.get_mut(self.node).unwrap()
    }
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
        child_trees: ArrayVec<InnerSumTree<T>, { 2 * TREE_BASE }>,
    },
    Leaf {
        items: ArrayVec<T, { 2 * TREE_BASE }>,
    },
}

impl<T: Item + GetSize> GetSize for Node<T>
where
    T::Summary: GetSize,
{
    fn get_heap_size(&self) -> usize {
        match self {
            Node::Internal { child_trees, .. } => {
                child_trees.iter().map(|tree| tree.get_heap_size()).sum()
            }
            Node::Leaf { items } => items.iter().map(|item| item.get_heap_size()).sum(),
        }
    }
}

impl<T> fmt::Debug for NodeRef<'_, T>
where
    T: Item + fmt::Debug,
    T::Summary: fmt::Debug + Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &**self {
            Node::Internal { child_trees } => f
                .debug_struct("Internal")
                .field("summary", &self.summary)
                .field(
                    "children",
                    &child_trees
                        .iter()
                        .map(|tree| NodeRef {
                            arena: self.arena,
                            node: tree.node,
                            summary: tree.summary,
                        })
                        .collect::<ArrayVec<_, { 2 * TREE_BASE }>>(),
                )
                .finish(),
            Node::Leaf { items } => f.debug_struct("Leaf").field("items", items).finish(),
        }
    }
}

impl<T: Item> Node<T> {
    fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf { .. })
    }

    fn summary(&self) -> T::Summary
    where
        T::Summary: Copy + Add<Output = T::Summary> + Min,
    {
        match self {
            Node::Internal { child_trees, .. } => sum(child_trees.iter().map(|t| t.summary)),
            Node::Leaf { items } => sum(items.iter().map(|item| item.summary())),
        }
    }

    fn into_child_trees(self) -> ArrayVec<InnerSumTree<T>, { 2 * TREE_BASE }> {
        match self {
            Node::Internal { child_trees, .. } => child_trees,
            Node::Leaf { .. } => panic!("Leaf nodes have no child trees"),
        }
    }

    #[cfg(test)]
    fn child_trees(&self) -> &ArrayVec<InnerSumTree<T>, { 2 * TREE_BASE }> {
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

fn sum<T, I>(iter: I) -> T
where
    T: Add<Output = T> + Min,
    I: IntoIterator<Item = T>,
{
    iter.into_iter().fold(T::MIN, |sum, value| sum + value)
}

pub struct End;

impl<D> Equivalent<D> for End {
    fn equivalent(&self, _: &D) -> bool {
        false
    }
}

impl<D> Comparable<D> for End {
    fn compare(&self, _: &D) -> std::cmp::Ordering {
        std::cmp::Ordering::Greater
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equivalent::{Comparable, Equivalent};
    use rand::{distr, prelude::*};
    use std::cmp;

    #[test]
    fn test_extend_and_push_tree() {
        let mut arena = Arena::default();
        let mut tree1 = SumTree::new(&mut arena);

        tree1 = tree1.push(0, &mut arena);
        tree1 = tree1.push(1, &mut arena);
        tree1 = tree1.push(2, &mut arena);
        tree1 = tree1.push(3, &mut arena);
        tree1 = tree1.push(4, &mut arena);
        tree1 = tree1.push(5, &mut arena);
        tree1 = tree1.push(6, &mut arena);

        tree1 = tree1.extend(7..20, &mut arena);

        let mut tree2 = SumTree::new(&mut arena);
        tree2 = tree2.extend(50..100, &mut arena);

        tree1 = tree1.append(tree2, &mut arena);

        assert_eq!(
            tree1.items(&arena),
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
            let mut tree = SumTree::<u8>::new(&mut arena);
            let count = rng.random_range(0..10);
            tree = tree.extend(
                rng.sample_iter(distr::StandardUniform).take(count),
                &mut arena,
            );

            for _ in 0..num_operations {
                let splice_end = rng.random_range(0..tree.extent::<Count>().0 + 1);
                let splice_start = rng.random_range(0..splice_end + 1);
                let count = rng.random_range(0..10);
                let tree_end = tree.extent::<Count>();
                let new_items = rng
                    .sample_iter(distr::StandardUniform)
                    .take(count)
                    .collect::<Vec<u8>>();

                let mut reference_items = tree.items(&arena);
                reference_items.splice(splice_start..splice_end, new_items.clone());

                tree = {
                    let mut cursor = tree.into_cursor::<Count>(&mut arena);
                    let mut new_tree = cursor.slice(&Count(splice_start), Bias::Right, &mut arena);
                    new_tree = new_tree.extend(new_items, &mut arena);
                    let _ = cursor.slice(&Count(splice_end), Bias::Right, &mut arena);
                    new_tree.append(cursor.slice(&tree_end, Bias::Right, &mut arena), &mut arena)
                };

                assert_eq!(tree.items(&arena), reference_items);
                // assert_eq!(
                //     tree.iter().collect::<Vec<_>>(),
                //     tree.cursor::<()>().collect::<Vec<_>>()
                // );

                let mut before_start = false;
                let mut cursor = tree.cursor::<Count>(&arena);
                let start_pos = rng.random_range(0..=reference_items.len());
                cursor.seek_forward(&Count(start_pos), Bias::Right, &arena);
                let mut pos = rng.random_range(start_pos..=reference_items.len());
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
                        cursor.next(&arena);
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
            //     let end = rng.random_range(0..tree.extent::<Count>().0 + 1);
            //     let start = rng.random_range(0..end + 1);
            //     let start_bias = if rng.r#random() { Bias::Left } else { Bias::Right };
            //     let end_bias = if rng.r#random() { Bias::Left } else { Bias::Right };

            //     let mut cursor = tree.cursor::<Count>();
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
        let tree = SumTree::<u8>::new(&mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&arena);
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
        drop(cursor);

        // Single-element tree
        let tree = SumTree::<u8>::new(&mut arena).extend(vec![1], &mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&arena);
        cursor.seek_forward(&Count(0), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 0);

        cursor.next(&arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 1);

        // cursor.prev();
        // assert_eq!(cursor.item(), Some(&1));
        // assert_eq!(cursor.prev_item(), None);
        // assert_eq!(cursor.next_item(&arena), None);
        // assert_eq!(cursor.start().sum, 0);
        drop(cursor);

        let mut cursor = tree.cursor::<IntegersSummary>(&arena);
        cursor.seek_forward(&tree.extent::<Count>(), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 1);

        cursor.seek(&Count(0), Bias::Right, &arena);
        cursor.seek_forward(&tree.extent::<Count>(), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&1));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 1);
        drop(cursor);

        // Multiple-element tree
        let tree = SumTree::new(&mut arena).extend(vec![1, 2, 3, 4, 5, 6], &mut arena);
        let mut cursor = tree.cursor::<IntegersSummary>(&arena);

        cursor.seek_forward(&Count(2), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), Some(&3));
        // assert_eq!(cursor.prev_item(), Some(&2));
        assert_eq!(cursor.next_item(&arena), Some(&4));
        assert_eq!(cursor.start().sum, 3);

        cursor.next(&arena);
        assert_eq!(cursor.item(&arena), Some(&4));
        // assert_eq!(cursor.prev_item(), Some(&3));
        assert_eq!(cursor.next_item(&arena), Some(&5));
        assert_eq!(cursor.start().sum, 6);

        cursor.next(&arena);
        assert_eq!(cursor.item(&arena), Some(&5));
        // assert_eq!(cursor.prev_item(), Some(&4));
        assert_eq!(cursor.next_item(&arena), Some(&6));
        assert_eq!(cursor.start().sum, 10);

        cursor.next(&arena);
        assert_eq!(cursor.item(&arena), Some(&6));
        // assert_eq!(cursor.prev_item(), Some(&5));
        assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 15);

        cursor.next(&arena);
        cursor.next(&arena);
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
        drop(cursor);

        let mut cursor = tree.cursor::<IntegersSummary>(&arena);
        cursor.seek_forward(&tree.extent::<Count>(), Bias::Right, &arena);
        assert_eq!(cursor.item(&arena), None);
        // assert_eq!(cursor.prev_item(), Some(&6));
        // assert_eq!(cursor.next_item(&arena), None);
        assert_eq!(cursor.start().sum, 21);

        cursor.seek(&Count(3), Bias::Right, &arena);
        cursor.seek_forward(&tree.extent::<Count>(), Bias::Right, &arena);
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
        drop(cursor);
    }

    #[test]
    fn test_from_iter() {
        let mut arena = Arena::default();
        assert_eq!(
            SumTree::new(&mut arena)
                .extend(0..100, &mut arena)
                .items(&arena),
            (0..100).collect::<Vec<_>>(),
        );

        // Ensure `from_iter` works correctly when the given iterator restarts
        // after calling `next` if `None` was already returned.
        let mut ix = 0;
        let iterator = std::iter::from_fn(|| {
            ix = (ix + 1) % 2;
            if ix == 1 { Some(1) } else { None }
        });
        assert_eq!(
            SumTree::new(&mut arena)
                .extend(iterator, &mut arena)
                .items(&arena),
            vec![1]
        );
    }

    #[derive(Copy, Clone, Debug)]
    pub struct IntegersSummary {
        count: usize,
        sum: usize,
        contains_even: bool,
        max: u8,
    }

    #[derive(Copy, Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
    struct Count(usize);

    #[derive(Copy, Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
    struct Sum(usize);

    #[derive(Copy, Ord, PartialOrd, Eq, PartialEq, Clone, Debug)]
    struct Max(u8);

    impl Item for u8 {
        type Summary = IntegersSummary;

        fn summary(&self) -> Self::Summary {
            IntegersSummary {
                count: 1,
                sum: *self as usize,
                contains_even: (*self & 1) == 0,
                max: *self,
            }
        }
    }

    impl Min for IntegersSummary {
        const MIN: Self = IntegersSummary {
            count: 0,
            sum: 0,
            contains_even: false,
            max: 0,
        };
    }

    impl Add for IntegersSummary {
        type Output = Self;

        fn add(mut self, other: Self) -> Self {
            self.count += other.count;
            self.sum += other.sum;
            self.contains_even |= other.contains_even;
            self.max = cmp::max(self.max, other.max);
            self
        }
    }

    impl Min for Max {
        const MIN: Self = Max(0);
    }

    impl Add<IntegersSummary> for Max {
        type Output = Self;

        fn add(self, rhs: IntegersSummary) -> Self::Output {
            Max(rhs.max)
        }
    }

    impl Min for Count {
        const MIN: Self = Count(0);
    }

    impl Add<IntegersSummary> for Count {
        type Output = Self;

        fn add(self, rhs: IntegersSummary) -> Self::Output {
            Count(self.0 + rhs.count)
        }
    }

    impl Equivalent<IntegersSummary> for Count {
        fn equivalent(&self, key: &IntegersSummary) -> bool {
            self.0 == key.count
        }
    }

    impl Comparable<IntegersSummary> for Count {
        fn compare(&self, cursor_location: &IntegersSummary) -> std::cmp::Ordering {
            std::cmp::Ord::cmp(&self.0, &cursor_location.count)
        }
    }

    impl Min for Sum {
        const MIN: Self = Sum(0);
    }

    impl Add<IntegersSummary> for Sum {
        type Output = Self;

        fn add(self, rhs: IntegersSummary) -> Self::Output {
            Sum(self.0 + rhs.sum)
        }
    }
}
