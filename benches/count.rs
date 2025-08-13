use std::{collections::{BTreeMap, HashMap}, ops::Range};

use count_range_sketch::CountRangeSketch;
use rand::{Rng, SeedableRng, rngs::SmallRng};

criterion::criterion_main!(benches);
criterion::criterion_group!(benches, counts);

const N: usize = 10000;
const RANGE: Range<u64> = 0..100000;

fn counts(c: &mut criterion::Criterion) {
    let seed = rand::random();

    {
        let mut g = c.benchmark_group("count");
        g.bench_function("sketch", |b| {
            let mut sketch = CountRangeSketch::new(N);
            let mut rng = SmallRng::from_seed(seed);

            b.iter_batched(
                || rng.random_range(RANGE),
                |n: u64| sketch.count(n),
                criterion::BatchSize::SmallInput,
            );
        });

        g.bench_function("btreemap", |b| {
            let mut map = BTreeMap::<u64, usize>::new();
            let mut rng = SmallRng::from_seed(seed);

            b.iter_batched(
                || rng.random_range(RANGE),
                |n: u64| *map.entry(n).or_default() += 1,
                criterion::BatchSize::SmallInput,
            );
        });

        g.bench_function("hashmap", |b| {
            let mut map =
                HashMap::<u64, usize, _>::with_hasher(foldhash::fast::RandomState::default());
            let mut rng = SmallRng::from_seed(seed);

            b.iter_batched(
                || rng.random_range(RANGE),
                |n: u64| *map.entry(n).or_default() += 1,
                criterion::BatchSize::SmallInput,
            );
        });
        g.finish();
    }

    {
        let mut g = c.benchmark_group("get");

        g.bench_function("sketch", |b| {
            let mut sketch = CountRangeSketch::new(N);
            let mut rng = SmallRng::from_seed(seed);

            for _ in RANGE {
                sketch.count(rng.random_range(RANGE));
            }

            b.iter_batched(
                || rng.random_range(RANGE),
                |n: u64| sketch.get(n..=n),
                criterion::BatchSize::SmallInput,
            );
        });

        g.bench_function("btreemap", |b| {
            let mut map = BTreeMap::<u64, usize>::new();
            let mut rng = SmallRng::from_seed(seed);

            for _ in RANGE {
                *map.entry(rng.random_range(RANGE)).or_default() += 1
            }

            b.iter_batched(
                || rng.random_range(RANGE),
                |n: u64| map.get(&n).copied().unwrap_or_default(),
                criterion::BatchSize::SmallInput,
            );
        });

        g.bench_function("hashmap", |b| {
            let mut map =
                HashMap::<u64, usize, _>::with_hasher(foldhash::fast::RandomState::default());
            let mut rng = SmallRng::from_seed(seed);

            for _ in RANGE {
                *map.entry(rng.random_range(RANGE)).or_default() += 1
            }

            b.iter_batched(
                || rng.random_range(RANGE),
                |n: u64| map.get(&n).copied().unwrap_or_default(),
                criterion::BatchSize::SmallInput,
            );
        });
    }
}
