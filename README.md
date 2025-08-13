# Count Range Sketch

A `CountRangeSketch<T>` allows counting the occurence of some `T`, for instance IP addresses, but will dynamically
merge nearby entries when the number of unique entries exceeds some set limit.

## Example

```rust
// we can only store 4 entries in this sketch.
let mut sketch = count_range_sketch::CountRangeSketch::new(4);

sketch.count(1);
sketch.count(1);
sketch.count(1);
sketch.count(1);

sketch.count(2);
sketch.count(2);

sketch.count(3);

sketch.count(5);
sketch.count(5);
sketch.count(5);

// All counts are as precise as they can be
assert_eq!(sketch.get(1..=1), (1..=1, 4));
assert_eq!(sketch.get(2..=2), (2..=2, 2));
assert_eq!(sketch.get(3..=3), (3..=3, 1));
assert_eq!(sketch.get(4..=4), (4..=4, 0));
assert_eq!(sketch.get(5..=5), (5..=5, 3));

// we can request different ranges
assert_eq!(sketch.get(..3), (1..=2, 6));
assert_eq!(sketch.get(3..), (3..=5, 4));
assert_eq!(sketch.get(..), (1..=5, 10));

assert_eq!(sketch.len(), 4);
// get all the occupied ranges
assert_eq!(
    sketch.get_all(), 
    vec![
        (1..=1, 4),
        (2..=2, 2),
        (3..=3, 1),
        (5..=5, 3),
    ],
);

// The counts will merge when exceeding the limit.
sketch.count(4);
assert_eq!(sketch.get(1..=1), (1..=1, 4));
assert_eq!(sketch.get(2..=2), (2..=2, 2));
assert_eq!(sketch.get(3..=3), (3..=5, 5));
assert_eq!(sketch.get(4..=4), (3..=5, 5));
assert_eq!(sketch.get(5..=5), (3..=5, 5));

assert_eq!(sketch.get(..3), (1..=2, 6));
assert_eq!(sketch.get(3..), (3..=5, 5));
assert_eq!(sketch.get(..), (1..=5, 11));

assert_eq!(sketch.len(), 3);
// get all the occupied ranges
assert_eq!(
    sketch.get_all(), 
    vec![
        (1..=1, 4),
        (2..=2, 2),
        (3..=5, 5),
    ],
);
```
