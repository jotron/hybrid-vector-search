use ordered_float::NotNan;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hybrid_vector_search::K_NEAREST_NEIGHBOURS;
use hybrid_vector_search::QUERY_DIMENSIONS;

fn main() {
    hybrid_vector_search::run_with_solver(solve);
}

pub fn solve(nodes: Vec<Vec<f32>>, queries: Vec<Vec<f32>>) -> Vec<Vec<u32>> {
    let mut result = vec![vec![0u32; K_NEAREST_NEIGHBOURS]; queries.len()];

    // Pre-Sorting
    let node_values: Vec<i32> = nodes.iter().map(|x| x[0] as i32).collect();
    let node_timestamps: Vec<f32> = nodes.iter().map(|x| x[1]).collect();
    let mut node_ids_sorted_by_timestamp = Vec::from_iter(0..nodes.len());
    node_ids_sorted_by_timestamp.sort_by_key(|i| NotNan::new(node_timestamps[*i]).unwrap());
    let mut node_ids_sorted_by_value_timestamp = Vec::from_iter(0..nodes.len());
    node_ids_sorted_by_value_timestamp
        .sort_by_key(|i| (node_values[*i], NotNan::new(node_timestamps[*i]).unwrap()));

    result
        .par_iter_mut()
        .enumerate()
        .zip(&queries)
        .for_each(|((i, out), query)| {
            let query_type: u32 = query[0] as u32;
            let v = query[1] as i32;
            let l = query[2];
            let r = query[3];
            let vec = &query[4..QUERY_DIMENSIONS];

            // To remember closest nodes so far
            let mut pq: BinaryHeap<(NotNan<f32>, u32)> = BinaryHeap::new();

            // Type 0 query
            if query_type == 0 {
                for j in 0..nodes.len() {
                    let base_vec = &nodes[j][2..]; // skip first 2 dimensions
                    let dist = NotNan::new(normal_l2(base_vec, vec)).unwrap();
                    pq.push((-dist, j as u32))
                }
            }
            // Type 2 query (timestamp)
            else if query_type == 2 {
                let (start_id, end_id) = (
                    node_ids_sorted_by_timestamp
                        .binary_search_by(|i| match node_timestamps[*i].partial_cmp(&l) {
                            Some(Ordering::Equal) => Ordering::Greater,
                            Some(ord) => ord,
                            None => Ordering::Equal,
                        })
                        .unwrap_err(),
                    node_ids_sorted_by_timestamp
                        .binary_search_by(|i| match node_timestamps[*i].partial_cmp(&r) {
                            Some(Ordering::Equal) => Ordering::Less,
                            Some(ord) => ord,
                            None => Ordering::Equal,
                        })
                        .unwrap_err(),
                );

                for j in start_id..end_id {
                    let base_vec = &nodes[node_ids_sorted_by_timestamp[j]][2..]; // skip first 2 dimensions
                    let dist = NotNan::new(normal_l2(base_vec, vec)).unwrap();
                    pq.push((-dist, node_ids_sorted_by_timestamp[j] as u32))
                }
            }
            // Type 1 & 3 query (timestamp)
            else {
                let (start_id, end_id) = match query_type {
                    // Value Constraint
                    1 => (
                        node_ids_sorted_by_value_timestamp
                            .binary_search_by(|i| match node_values[*i].cmp(&v) {
                                Ordering::Equal => Ordering::Greater,
                                ord => ord,
                            })
                            .unwrap_err(),
                        node_ids_sorted_by_value_timestamp
                            .binary_search_by(|i| match node_values[*i].cmp(&v) {
                                Ordering::Equal => Ordering::Less,
                                ord => ord,
                            })
                            .unwrap_err(),
                    ),
                    // Time & Value Constraint
                    3 => (
                        node_ids_sorted_by_value_timestamp
                            .binary_search_by(|i| {
                                match (node_values[*i], node_timestamps[*i]).partial_cmp(&(v, l)) {
                                    Some(Ordering::Equal) => Ordering::Greater,
                                    Some(ord) => ord,
                                    None => Ordering::Equal,
                                }
                            })
                            .unwrap_err(),
                        node_ids_sorted_by_value_timestamp
                            .binary_search_by(|i| {
                                match (node_values[*i], node_timestamps[*i]).partial_cmp(&(v, r)) {
                                    Some(Ordering::Equal) => Ordering::Less,
                                    Some(ord) => ord,
                                    None => Ordering::Equal,
                                }
                            })
                            .unwrap_err(),
                    ),
                    _ => panic!(),
                };

                for j in start_id..end_id {
                    let base_vec = &nodes[node_ids_sorted_by_value_timestamp[j]][2..]; // skip first 2 dimensions
                    let dist = NotNan::new(normal_l2(base_vec, vec)).unwrap();
                    pq.push((-dist, node_ids_sorted_by_value_timestamp[j] as u32))
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

            if i % 100 == 0 {
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
