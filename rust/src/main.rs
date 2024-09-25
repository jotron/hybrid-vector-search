use std::collections::BinaryHeap;
use std::env;
use std::time::Instant;
use std::cmp::Ordering;

const VECTOR_DIMENSIONS: u32 = 100;
const NODE_DIMENSIONS: u32 = 102;
const QUERY_DIMENSIONS: u32 = 104;
const K_NUM_NEAREST_NEIGHBOUR = 100;

struct Pair {
    dist: f32,
    index: u32,
}
impl Eq for Pair {}
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max-heap behavior (since we need min-heap)
        other.dist.partial_cmp(&self.dist).unwrap()
    }
}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let source_path = &args[1];
    let query_path = &args[2];
    let output_path = &args[3];

    let now = Instant::now();

    // Read nodes & queries
    let mut nodes: Vec<Vec<f32>>;
    let mut queries: Vec<Vec<f32>>;
    read_bin(source_path, NODE_DIMENSIONS, &mut nodes);
    read_bin(query_path, QUERY_DIMENSIONS, &mut queries);

    // Generate ground truth
    let mut knns: Vec<Vec<u32>>;
    bruteforce(&nodes, &queries, &mut knns);

    // Compare to ground truth on disk to calculate recall.
    let mut gt_knns: Vec<Vec<u32>>;
    read_output_bin("../../data/label.bin", &mut gt_knns, queries.len());
    let recall = get_knn_recall(knns, gt_knns);
    println!("Recall: {}", recall);

    println!("Total time: {}", now.elapsed().as_secs());
}

fn brute_force(nodes: &[Vec<f32>], queries: &[Vec<f32>], knns: &mut Vec<Vec<u32>>) {
    let n = nodes.len();
    let nq = queries.len();

    knns.resize(nq, Vec::new());

    for (i, query) in queries.iter().enumerate() {
        let query_type = query[0] as u32;
        let v = query[1] as i32;
        let l = query[2];
        let r = query[3];
        let vec = &query[4..];

        let mut pq = BinaryHeap::new();

        for (j, node) in nodes.iter().enumerate() {
            let node_vec = &node[2..]; // skip first 2 dimensions
            let node_val = node[0] as i32;
            let node_time = node[1];

            let dist = match query_type {
                0 => Some(normal_l2(node_vec, vec)),
                1 if node_val == v => Some(normal_l2(node_vec, vec)),
                2 if node_time >= l && node_time <= r => Some(normal_l2(node_vec, vec)),
                3 if v == node_val && node_time >= l && node_time <= r => {
                    Some(normal_l2(node_vec, vec))
                }
                _ => None,
            };

            pq.push(Pair {
                dist: -dist,
                index: j as u32,
            });
        }

        knns[i].resize(K_NUM_NEAREST_NEIGHBOUR, 0);
        for j in (0..K_NUM_NEAREST_NEIGHBOUR).rev() {
            let res = pq.pop().unwrap();
            knns[i][j] = res.index;
        }
    }
}

fn normal_l2(base_vec: &[f32], query_vec: &[f32], d: usize) -> f32 {
    base_vec.iter().zip(query_vec).take(d).map(|(b, q)| (b - q).powi(2)).sum()
}
