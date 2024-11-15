use ordered_float::NotNan;
use std::collections::BinaryHeap;

use hybrid_vector_search::K_NEAREST_NEIGHBOURS;
use hybrid_vector_search::QUERY_DIMENSIONS;

fn main() {
    hybrid_vector_search::run_with_solver(solve);
}

pub fn solve(nodes: Vec<Vec<f32>>, queries: Vec<Vec<f32>>) -> Vec<Vec<u32>> {
    let mut result = vec![vec![0u32; K_NEAREST_NEIGHBOURS]; queries.len()];

    for i in 0..queries.len() {
        let query_type = queries[i][0] as u32;
        let v = queries[i][1] as i32;
        let l = queries[i][2];
        let r = queries[i][3];
        let vec = &queries[i][4..QUERY_DIMENSIONS];

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
                result[i][j] = index;
            }
        }
    }

    return result;
}

fn normal_l2(base_vec: &[f32], query_vec: &[f32]) -> f32 {
    base_vec
        .iter()
        .zip(query_vec)
        .map(|(b, q)| (b - q).powi(2))
        .sum()
}
