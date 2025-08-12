use super::*;
use arrayvec::ArrayVec;
use std::{cmp::Ordering, mem};

#[derive(Clone)]
struct StackEntry<'a, T: Item, D> {
    tree: &'a SumTree<T>,
    index: usize,
    position: D,
}

impl<T: Item + fmt::Debug, D: fmt::Debug> fmt::Debug for StackEntry<'_, T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StackEntry")
            .field("index", &self.index)
            .field("position", &self.position)
            .finish()
    }
}

pub struct Cursor<'a, T: Item, D> {
    tree: &'a SumTree<T>,
    stack: ArrayVec<StackEntry<'a, T, D>, 16>,
    position: D,
    did_seek: bool,
    at_end: bool,
    cx: &'a <T::Summary as Summary>::Context,
}

// impl<T: Item + fmt::Debug, D: fmt::Debug> fmt::Debug for Cursor<'_, T, D>
// where
//     T::Summary: fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("Cursor")
//             .field("tree", &self.tree)
//             .field("stack", &self.stack)
//             .field("position", &self.position)
//             .field("did_seek", &self.did_seek)
//             .field("at_end", &self.at_end)
//             .finish()
//     }
// }

impl<'a, T, D> Cursor<'a, T, D>
where
    T: Item,
    D: Dimension<T::Summary>,
{
    pub fn new(
        tree: &'a SumTree<T>,
        cx: &'a <T::Summary as Summary>::Context,
        arena: &Arena<T>,
    ) -> Self {
        Self {
            stack: ArrayVec::new(),
            position: D::zero(cx),
            did_seek: false,
            at_end: tree.is_empty(arena),
            tree,
            cx,
        }
    }

    #[cfg(test)]
    fn reset(&mut self, arena: &Arena<T>) {
        self.did_seek = false;
        self.at_end = self.tree.is_empty(arena);
        self.stack.truncate(0);
        self.position = D::zero(self.cx);
    }

    #[cfg(test)]
    pub fn start(&self) -> &D {
        &self.position
    }

    // #[cfg(test)]
    // #[track_caller]
    // pub fn end(&self) -> D {
    //     if let Some(item_summary) = self.item_summary() {
    //         let mut end = *self.start();
    //         end.add_summary(item_summary, self.cx);
    //         end
    //     } else {
    //         *self.start()
    //     }
    // }

    /// Item is None, when the list is empty, or this cursor is at the end of the list.
    #[track_caller]
    pub fn item(&self, arena: &'a Arena<T>) -> Option<&'a T> {
        self.assert_did_seek();
        if let Some(entry) = self.stack.last() {
            match *entry.tree.get(arena).into() {
                Node::Leaf { ref items, .. } => {
                    if entry.index == items.len() {
                        None
                    } else {
                        Some(&items[entry.index])
                    }
                }
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    #[cfg(test)]
    #[track_caller]
    pub fn next_item(&self, arena: &'a Arena<T>) -> Option<&'a T> {
        self.assert_did_seek();
        if let Some(entry) = self.stack.last() {
            if entry.index == entry.tree.get(arena).items().len() - 1 {
                if let Some(next_leaf) = self.next_leaf(arena) {
                    Some(next_leaf.into().items().first().unwrap())
                } else {
                    None
                }
            } else {
                match *entry.tree.get(arena).into() {
                    Node::Leaf { ref items, .. } => Some(&items[entry.index + 1]),
                    _ => unreachable!(),
                }
            }
        } else if self.at_end {
            None
        } else {
            self.tree.first(arena)
        }
    }

    #[cfg(test)]
    #[track_caller]
    fn next_leaf(&self, arena: &'a Arena<T>) -> Option<NodeRef<'a, T>> {
        for entry in self.stack.iter().rev().skip(1) {
            if entry.index < entry.tree.get(arena).into().child_trees().len() - 1 {
                match *entry.tree.get(arena).into() {
                    Node::Internal {
                        ref child_trees, ..
                    } => return Some(child_trees[entry.index + 1].leftmost_leaf(arena)),
                    Node::Leaf { .. } => unreachable!(),
                };
            }
        }
        None
    }

    #[track_caller]
    pub fn next<'b: 'a>(&mut self, arena: &'b Arena<T>) {
        self.search_forward(|_| true, arena)
    }

    #[track_caller]
    pub fn search_forward<'b: 'a, F>(&mut self, mut filter_node: F, arena: &'b Arena<T>)
    where
        F: FnMut(T::Summary) -> bool,
    {
        let mut descend = false;

        if self.stack.is_empty() {
            if !self.at_end {
                self.stack.push(StackEntry {
                    tree: self.tree,
                    index: 0,
                    position: D::zero(self.cx),
                });
                descend = true;
            }
            self.did_seek = true;
        }

        while !self.stack.is_empty() {
            let new_subtree = {
                let entry = self.stack.last_mut().unwrap();
                match entry.tree.get(arena).into() {
                    Node::Internal { child_trees, .. } => {
                        if !descend {
                            entry.index += 1;
                            entry.position = self.position;
                        }

                        while entry.index < child_trees.len() {
                            let next_summary = child_trees[entry.index].summary;
                            if filter_node(next_summary) {
                                break;
                            } else {
                                entry.index += 1;
                                entry.position.add_summary(next_summary, self.cx);
                                self.position.add_summary(next_summary, self.cx);
                            }
                        }

                        child_trees.get(entry.index)
                    }
                    Node::Leaf { items } => {
                        if !descend {
                            let item_summary = items[entry.index].summary(self.cx);
                            entry.index += 1;
                            entry.position.add_summary(item_summary, self.cx);
                            self.position.add_summary(item_summary, self.cx);
                        }

                        loop {
                            if let Some(item) = items.get(entry.index) {
                                let next_item_summary = item.summary(self.cx);
                                if filter_node(next_item_summary) {
                                    return;
                                } else {
                                    entry.index += 1;
                                    entry.position.add_summary(next_item_summary, self.cx);
                                    self.position.add_summary(next_item_summary, self.cx);
                                }
                            } else {
                                break None;
                            }
                        }
                    }
                }
            };

            if let Some(subtree) = new_subtree {
                descend = true;
                self.stack.push(StackEntry {
                    tree: subtree,
                    index: 0,
                    position: self.position,
                });
            } else {
                descend = false;
                self.stack.pop();
            }
        }

        self.at_end = self.stack.is_empty();
        debug_assert!(
            self.stack.is_empty() || self.stack.last().unwrap().tree.get(arena).is_leaf()
        );
    }

    #[track_caller]
    fn assert_did_seek(&self) {
        assert!(
            self.did_seek,
            "Must call `seek`, `next` or `prev` before calling this method"
        );
    }

    #[cfg(test)]
    #[track_caller]
    pub fn seek<Target>(&mut self, pos: &Target, bias: Bias, arena: &'a Arena<T>)
    where
        Target: SeekTarget<T::Summary, D>,
    {
        self.reset(arena);
        self.seek_forward(pos, bias, arena)
    }

    #[track_caller]
    pub fn seek_forward<Target>(&mut self, target: &Target, bias: Bias, arena: &'a Arena<T>)
    where
        Target: SeekTarget<T::Summary, D>,
    {
        assert!(
            target.cmp(&self.position, self.cx) >= Ordering::Equal,
            "cannot seek backward",
        );

        if !self.did_seek {
            self.did_seek = true;
            self.stack.push(StackEntry {
                tree: self.tree,
                index: 0,
                position: D::zero(self.cx),
            });
        }

        let mut ascending = false;
        'outer: while let Some(entry) = self.stack.last_mut() {
            match entry.tree.get(arena).into() {
                Node::Internal { child_trees, .. } => {
                    if ascending {
                        entry.index += 1;
                        entry.position = self.position;
                    }

                    for child in &child_trees[entry.index..] {
                        let mut child_end = self.position;
                        child_end.add_summary(child.summary, self.cx);

                        let comparison = target.cmp(&child_end, self.cx);
                        if comparison == Ordering::Greater
                            || (comparison == Ordering::Equal && bias == Bias::Right)
                        {
                            self.position = child_end;
                            entry.index += 1;
                            entry.position = self.position;
                        } else {
                            self.stack.push(StackEntry {
                                tree: child,
                                index: 0,
                                position: self.position,
                            });
                            ascending = false;
                            continue 'outer;
                        }
                    }
                }
                Node::Leaf { items } => {
                    for item in &items[entry.index..] {
                        let mut child_end = self.position;
                        child_end.add_summary(item.summary(self.cx), self.cx);

                        let comparison = target.cmp(&child_end, self.cx);
                        if comparison == Ordering::Greater
                            || (comparison == Ordering::Equal && bias == Bias::Right)
                        {
                            self.position = child_end;
                            entry.index += 1;
                        } else {
                            break 'outer;
                        }
                    }
                }
            }

            self.stack.pop();
            ascending = true;
        }

        self.at_end = self.stack.is_empty();
        debug_assert!(
            self.stack.is_empty() || self.stack.last().unwrap().tree.get(arena).is_leaf()
        );
    }
}

struct StackEntryOwned<T: Item, D> {
    tree: SumTree<T>,
    position: D,
}

impl<T: Item + fmt::Debug, D: fmt::Debug> fmt::Debug for StackEntryOwned<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StackEntryOwned")
            .field("position", &self.position)
            .finish()
    }
}

pub struct IntoCursor<'a, T: Item, D> {
    stack: ArrayVec<StackEntryOwned<T, D>, 16>,
    position: D,
    cx: &'a <T::Summary as Summary>::Context,
}

impl<'a, T, D> IntoCursor<'a, T, D>
where
    T: Item,
    D: Dimension<T::Summary>,
{
    pub fn new(tree: SumTree<T>, cx: &'a <T::Summary as Summary>::Context) -> Self {
        let mut stack = ArrayVec::new();

        stack.push(StackEntryOwned {
            tree,
            position: D::zero(cx),
        });

        Self {
            stack,
            position: D::zero(cx),
            cx,
        }
    }

    /// Advances the cursor and returns traversed items as a tree.
    pub fn slice<Target>(&mut self, end: &Target, bias: Bias, arena: &mut Arena<T>) -> SumTree<T>
    where
        Target: SeekTarget<T::Summary, D>,
    {
        let mut slice = SliceSeekAggregate {
            tree: SumTree::new(self.cx, arena),
            leaf_items: ArrayVec::new(),
            leaf_summary: <T::Summary as Summary>::zero(self.cx),
        };
        self.seek_internal(end, bias, &mut slice, arena);
        slice.tree
    }

    pub fn suffix(mut self, arena: &mut Arena<T>) -> SumTree<T> {
        let suffix = self.slice(&End::new(), Bias::Right, arena);
        debug_assert!(self.stack.is_empty());
        suffix
    }

    pub fn next(&mut self, arena: &mut Arena<T>) -> Option<T> {
        loop {
            let Some(entry) = self.stack.last_mut() else {
                break None;
            };
            match entry.tree.get_mut(arena).into() {
                Node::Internal { child_trees, .. } => {
                    debug_assert!(!child_trees.is_empty());

                    let child_tree = child_trees.remove(0);

                    self.stack.push(StackEntryOwned {
                        tree: child_tree,
                        position: self.position,
                    });
                }
                Node::Leaf { items } => {
                    if items.is_empty() {
                        break None;
                    }

                    let item = items.remove(0);
                    self.position.add_summary(item.summary(self.cx), self.cx);

                    break Some(item);
                }
            }
        }
    }

    /// Returns whether we found the item you were seeking for
    #[track_caller]
    fn seek_internal(
        &mut self,
        target: &dyn SeekTarget<T::Summary, D>,
        bias: Bias,
        aggregate: &mut SliceSeekAggregate<T>,
        arena: &mut Arena<T>,
    ) {
        assert!(
            target.cmp(&self.position, self.cx) >= Ordering::Equal,
            "cannot seek backward",
        );

        let mut ascending = false;
        'outer: while let Some(entry) = self.stack.last_mut() {
            let summary = entry.tree.summary;
            match arena.0.remove(entry.tree.node).unwrap() {
                Node::Internal {
                    height,
                    child_trees,
                } => {
                    if ascending {
                        entry.position = self.position;
                    }

                    let mut tree_iter = child_trees.into_iter();

                    loop {
                        let Some(child) = tree_iter.next() else { break };

                        let mut child_end = self.position;
                        child_end.add_summary(child.summary, self.cx);

                        let comparison = target.cmp(&child_end, self.cx);
                        if comparison == Ordering::Greater
                            || (comparison == Ordering::Equal && bias == Bias::Right)
                        {
                            self.position = child_end;
                            aggregate.push_tree(child, self.cx, arena);
                            entry.position = self.position;
                        } else {
                            entry.tree = arena.alloc(
                                summary,
                                Node::Internal {
                                    height,
                                    child_trees: tree_iter.collect(),
                                },
                            );

                            self.stack.push(StackEntryOwned {
                                tree: child,
                                position: self.position,
                            });
                            ascending = false;
                            continue 'outer;
                        }
                    }
                }
                Node::Leaf { items } => {
                    let mut items_iter = items.into_iter();
                    loop {
                        let Some(item) = items_iter.next() else { break };
                        let summary = item.summary(self.cx);

                        let mut child_end = self.position;
                        child_end.add_summary(summary, self.cx);

                        let comparison = target.cmp(&child_end, self.cx);
                        if comparison == Ordering::Greater
                            || (comparison == Ordering::Equal && bias == Bias::Right)
                        {
                            self.position = child_end;
                            aggregate.push_item(item, summary, self.cx);
                        } else {
                            let mut items = ArrayVec::new();

                            items.push(item);
                            items.extend(items_iter);
                            entry.tree = arena.alloc(summary, Node::Leaf { items });

                            aggregate.end_leaf(self.cx, arena);
                            break 'outer;
                        }
                    }

                    aggregate.end_leaf(self.cx, arena);
                }
            }

            self.stack.pop();
            ascending = true;
        }

        debug_assert!(
            self.stack.is_empty() || self.stack.last().unwrap().tree.get(arena).is_leaf()
        );
    }
}

struct SliceSeekAggregate<T: Item> {
    tree: SumTree<T>,
    leaf_items: ArrayVec<T, { 2 * TREE_BASE }>,
    leaf_summary: T::Summary,
}

impl<T: Item> SliceSeekAggregate<T> {
    fn end_leaf(&mut self, cx: &<T::Summary as Summary>::Context, arena: &mut Arena<T>) {
        let summary = mem::replace(&mut self.leaf_summary, <T::Summary as Summary>::zero(cx));
        let leaf = arena.alloc(
            summary,
            Node::Leaf {
                items: mem::take(&mut self.leaf_items),
            },
        );

        replace_with::replace_with_or_abort(&mut self.tree, |tree| tree.append(leaf, cx, arena));
    }

    fn push_item(&mut self, item: T, summary: T::Summary, cx: &<T::Summary as Summary>::Context) {
        self.leaf_items.push(item);
        Summary::add_summary(&mut self.leaf_summary, summary, cx);
    }

    fn push_tree(
        &mut self,
        tree: SumTree<T>,
        cx: &<T::Summary as Summary>::Context,
        arena: &mut Arena<T>,
    ) {
        replace_with::replace_with_or_abort(&mut self.tree, |agg| agg.append(tree, cx, arena));
    }
}

struct End<D>(PhantomData<D>);

impl<D> End<D> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl<S: Summary, D: Dimension<S>> SeekTarget<S, D> for End<D> {
    fn cmp(&self, _: &D, _: &S::Context) -> Ordering {
        Ordering::Greater
    }
}

impl<D> fmt::Debug for End<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("End").finish()
    }
}
