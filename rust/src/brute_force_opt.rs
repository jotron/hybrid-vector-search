use ordered_float::NotNan;
use rayon::prelude::*;
use std::collections::BinaryHeap;

const QUERY_DIMENSIONS: usize = 104;
const K_NEAREST_NEIGHBOURS: usize = 100;

pub fn solve(nodes: Vec<Vec<f32>>, queries: Vec<Vec<f32>>) -> Vec<Vec<u32>> {
    let mut result = vec![vec![0u32; K_NEAREST_NEIGHBOURS]; queries.len()];

    result
        .par_iter_mut()
        .enumerate()
        .zip(&queries)
        .for_each(|((i, out), query)| {
            let query_type = query[0] as u32;
            let v = query[1] as i32;
            let l = query[2];
            let r = query[3];
            let vec = &query[4..QUERY_DIMENSIONS];

            let mut pq: BinaryHeap<(NotNan<f32>, u32)> = BinaryHeap::new();

            for j in 0..nodes.len() {
                let base_vec = &nodes[j][2..]; // skip first 2 dimensions
                let bv = nodes[j][0] as i32;
                let bt = nodes[j][1];

                let dist = NotNan::new(normal_l2(base_vec, vec)).unwrap();

                match query_type {
                    0 => pq.push((-dist, j as u32)),
                    1 if v == bv => pq.push((-dist, j as u32)),
                    2 if bt >= l && bt <= r => pq.push((-dist, j as u32)),
                    3 if v == bv && bt >= l && bt <= r => pq.push((-dist, j as u32)),
                    _ => (),
                }
            }

            // Store
            if pq.len() < K_NEAREST_NEIGHBOURS {
                println!("id: {}", i);
                println!("query type: {} v: {} l: {} r: {}", query_type, v, l, r);
                println!("K: {}", pq.len());
            }
            for j in 0..K_NEAREST_NEIGHBOURS {
                if let Some((_dist, index)) = pq.pop() {
                    out[j] = index;
                }
            }

            if i % 1000 == 0 {
                println!("Processed {i}/{} queries", queries.len());
            }
        });

    return result;
}

fn normal_l2(base_vec: &[f32], query_vec: &[f32]) -> f32 {
    base_vec
        .iter()
        .zip(query_vec)
        .map(|(b, q)| (b - q).powi(2))
        .sum()
}
