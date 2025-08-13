use arrayvec::ArrayVec;
use equivalent::Comparable;
use std::{cmp::Ordering, fmt, ops::Add};

#[cfg(test)]
use crate::sumtree::NodeRef;
use crate::sumtree::{Arena, Bias, Empty, End, InnerSumTree, Item, Min, Node, SumTree, TREE_BASE};

#[derive(Clone)]
struct StackEntry<'a, T: Item, D> {
    tree: &'a InnerSumTree<T>,
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
}

impl<'a, T, D> Cursor<'a, T, D>
where
    T: Item,
    T::Summary: Copy,
    D: Add<T::Summary, Output = D> + Min + Copy,
{
    pub fn new(tree: &'a SumTree<T>, arena: &Arena<T>) -> Self {
        Self {
            stack: ArrayVec::new(),
            position: D::MIN,
            did_seek: false,
            at_end: tree.is_empty(arena),
            tree,
        }
    }

    #[cfg(test)]
    fn reset(&mut self, arena: &Arena<T>) {
        self.did_seek = false;
        self.at_end = self.tree.is_empty(arena);
        self.stack.truncate(0);
        self.position = D::MIN;
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
    //         end.add_summary(item_summary);
    //         end
    //     } else {
    //         *self.start()
    //     }
    // }

    /// Item is None, when the list is empty, or this cursor is at the end of the list.
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

    pub fn next<'b: 'a>(&mut self, arena: &'b Arena<T>) {
        self.search_forward(|_| true, arena)
    }

    pub fn search_forward<'b: 'a, F>(&mut self, mut filter_node: F, arena: &'b Arena<T>)
    where
        F: FnMut(T::Summary) -> bool,
    {
        let mut descend = false;

        if self.stack.is_empty() {
            if !self.at_end {
                self.stack.push(StackEntry {
                    tree: &self.tree.inner,
                    index: 0,
                    position: D::MIN,
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
                                entry.position = entry.position + next_summary;
                                self.position = self.position + next_summary;
                            }
                        }

                        child_trees.get(entry.index)
                    }
                    Node::Leaf { items } => {
                        if !descend {
                            let item_summary = items[entry.index].summary();
                            entry.index += 1;
                            entry.position = entry.position + item_summary;
                            self.position = self.position + item_summary;
                        }

                        loop {
                            if let Some(item) = items.get(entry.index) {
                                let next_item_summary = item.summary();
                                if filter_node(next_item_summary) {
                                    return;
                                } else {
                                    entry.index += 1;
                                    entry.position = entry.position + next_item_summary;
                                    self.position = self.position + next_item_summary;
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
    pub fn seek<Target>(&mut self, pos: &Target, bias: Bias, arena: &'a Arena<T>)
    where
        Target: Comparable<D>,
    {
        self.reset(arena);
        self.seek_forward(pos, bias, arena)
    }

    pub fn seek_forward<Target>(&mut self, target: &Target, bias: Bias, arena: &'a Arena<T>)
    where
        Target: Comparable<D>,
    {
        self.summary::<Target, Empty>(target, bias, arena);
    }

    pub fn summary<Target, Output>(
        &mut self,
        target: &Target,
        bias: Bias,
        arena: &'a Arena<T>,
    ) -> Output
    where
        Target: Comparable<D>,
        Output: Add<T::Summary, Output = Output> + Min + Copy,
    {
        assert!(
            target.compare(&self.position) >= Ordering::Equal,
            "cannot seek backward",
        );

        let mut output = Output::MIN;

        if !self.did_seek {
            self.did_seek = true;
            self.stack.push(StackEntry {
                tree: &self.tree.inner,
                index: 0,
                position: D::MIN,
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
                        let child_end = self.position + child.summary;

                        let comparison = target.compare(&child_end);
                        if comparison == Ordering::Greater
                            || (comparison == Ordering::Equal && bias == Bias::Right)
                        {
                            entry.index += 1;

                            output = output + child.summary;

                            self.position = child_end;
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
                        let item_summary = item.summary();
                        let child_end = self.position + item_summary;

                        let comparison = target.compare(&child_end);
                        if comparison == Ordering::Greater
                            || (comparison == Ordering::Equal && bias == Bias::Right)
                        {
                            entry.index += 1;

                            output = output + item_summary;

                            self.position = child_end;
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

        output
    }
}

enum StackEntryNode<T: Item, D> {
    Leaf {
        items: arrayvec::IntoIter<T, { 2 * TREE_BASE }>,
    },
    Internal {
        child_trees: arrayvec::IntoIter<InnerSumTree<T>, { 2 * TREE_BASE }>,
        position: D,
    },
}

impl<T: Item, D> StackEntryNode<T, D> {
    fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }
}

impl<T: Item> Node<T> {
    fn into_pos<D>(self, position: D) -> StackEntryNode<T, D> {
        match self {
            Node::Internal { child_trees } => StackEntryNode::Internal {
                child_trees: child_trees.into_iter(),
                position,
            },
            Node::Leaf { items } => StackEntryNode::Leaf {
                items: items.into_iter(),
            },
        }
    }
}

pub struct IntoCursor<T: Item, D> {
    height: u8,
    stack: ArrayVec<StackEntryNode<T, D>, 16>,
    position: D,
}

impl<T, D> IntoCursor<T, D>
where
    T: Item,
    T::Summary: Min + Copy + Add<Output = T::Summary>,
    D: Add<T::Summary, Output = D> + Min + Copy,
{
    pub fn new(tree: SumTree<T>, arena: &mut Arena<T>) -> Self {
        Self {
            height: tree.height,
            stack: ArrayVec::from_iter([arena.remove(tree.inner).into_pos(D::MIN)]),
            position: D::MIN,
        }
    }

    pub fn suffix(mut self, arena: &mut Arena<T>) -> SumTree<T> {
        let suffix = self.slice(&End, Bias::Right, arena);
        debug_assert!(self.stack.is_empty());
        suffix
    }

    pub fn next(&mut self, arena: &mut Arena<T>) -> Option<T> {
        loop {
            let Some(entry) = self.stack.last_mut() else {
                break None;
            };
            match entry {
                StackEntryNode::Internal { child_trees, .. } => {
                    let node = arena.remove(child_trees.next().unwrap());
                    self.stack.push(node.into_pos(self.position));
                }
                StackEntryNode::Leaf { items } => {
                    let item = items.next()?;
                    self.position = self.position + item.summary();

                    break Some(item);
                }
            }
        }
    }

    /// Advances the cursor and returns traversed items as a tree.
    pub fn slice<Target>(&mut self, target: &Target, bias: Bias, arena: &mut Arena<T>) -> SumTree<T>
    where
        Target: Comparable<D>,
    {
        let mut output = SumTree::new(arena);

        debug_assert!(
            target.compare(&self.position) >= Ordering::Equal,
            "cannot seek backward",
        );

        let mut ascending = false;
        'outer: while let Some(entry) = self.stack.last_mut() {
            match entry {
                StackEntryNode::Internal {
                    child_trees,
                    position,
                } => {
                    if ascending {
                        *position = self.position;
                    }

                    let mut i = 0;
                    for child in child_trees.as_slice() {
                        let end = self.position + child.summary;
                        match target.compare(&end) {
                            Ordering::Less => break,
                            Ordering::Equal if bias == Bias::Left => break,
                            _ => {}
                        }

                        i += 1;
                        self.position = end;
                    }

                    if i > 0 {
                        *position = self.position;
                        let node = Node::Internal {
                            child_trees: child_trees.by_ref().take(i).collect(),
                        };
                        output = output.append(
                            SumTree {
                                inner: arena.alloc(node.summary(), node),
                                height: self.height,
                            },
                            arena,
                        );
                    }

                    if let Some(child) = child_trees.next() {
                        let node = arena.remove(child);
                        self.stack.push(node.into_pos(self.position));
                        self.height -= 1;
                        ascending = false;
                        continue 'outer;
                    }
                }
                StackEntryNode::Leaf { items } => {
                    let mut i = 0;
                    for item in items.as_slice() {
                        let end = self.position + item.summary();
                        match target.compare(&end) {
                            Ordering::Less => break,
                            Ordering::Equal if bias == Bias::Left => break,
                            _ => {}
                        }

                        i += 1;
                        self.position = end;
                    }

                    if i > 0 {
                        let node = Node::Leaf {
                            items: items.by_ref().take(i).collect(),
                        };
                        output = output.append(
                            SumTree {
                                inner: arena.alloc(node.summary(), node),
                                height: self.height,
                            },
                            arena,
                        );
                    }

                    if items.len() != 0 {
                        break 'outer;
                    }
                }
            }

            self.stack.pop();
            self.height += 1;
            ascending = true;
        }

        debug_assert!(self.stack.is_empty() || self.stack.last().unwrap().is_leaf());

        output
    }
}
