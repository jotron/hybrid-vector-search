use ngt::{NgtIndex, NgtProperties, NgtDistance};
use ordered_float::NotNan;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const QUERY_DIMENSIONS: usize = 104;
const K_NEAREST_NEIGHBOURS: usize = 100;
const VECTOR_DIMENSION: usize = 100;

pub fn solve(nodes: Vec<Vec<f32>>, queries: Vec<Vec<f32>>) -> Vec<Vec<u32>> {
    let mut result = vec![vec![0u32; K_NEAREST_NEIGHBOURS]; queries.len()];

    // Pre-Sorting
    let node_values: Vec<i32> = nodes.iter().map(|x| x[0] as i32).collect();
    let node_timestamps: Vec<f32> = nodes.iter().map(|x| x[1]).collect();
    let node_vectors: Vec<Vec<f32>> = nodes.iter().map(|x| x[2..].to_vec()).collect();
    let mut node_ids_sorted_by_timestamp = Vec::from_iter(0..nodes.len());
    node_ids_sorted_by_timestamp.sort_by_key(|i| NotNan::new(node_timestamps[*i]).unwrap());
    let mut node_ids_sorted_by_value_timestamp = Vec::from_iter(0..nodes.len());
    node_ids_sorted_by_value_timestamp
        .sort_by_key(|i| (node_values[*i], NotNan::new(node_timestamps[*i]).unwrap()));

    // Create index
    let prop = NgtProperties::<f32>::dimension(VECTOR_DIMENSION).unwrap().distance_type(NgtDistance::L2).unwrap();
    let mut index: NgtIndex<f32> = NgtIndex::create("index", prop).unwrap();
    index.insert_batch(node_vectors).unwrap();
    index.build(8).unwrap(); // 8 Threads

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
                let res = index.search(&vec, K_NEAREST_NEIGHBOURS, 0.1).unwrap();
                for search_result in res {
                    pq.push((NotNan::new(0.0).unwrap(), search_result.id-1 as u32))
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
            //println!("id: {i} query type: {} v: {} l: {} r: {} K: {}", query_type, v, l, r, pq.len());
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
