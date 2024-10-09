use bytemuck;
use ordered_float::NotNan;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::time::Instant;

const VECTOR_DIMENSIONS: u32 = 100;
const NODE_DIMENSIONS: u32 = 102;
const QUERY_DIMENSIONS: u32 = 104;
const K_NEAREST_NEIGHBOURS: u32 = 100;

fn main() {
    let args: Vec<String> = env::args().collect();
    let source_path = &args[1];
    let query_path = &args[2];

    let start = Instant::now();

    // Read nodes & queries
    let nodes = read_bin(source_path, NODE_DIMENSIONS as usize);
    let queries = read_bin(query_path, QUERY_DIMENSIONS as usize);

    // Calculate
    let knns: Vec<Vec<u32>> = brute_force(&nodes, &queries);

    // Read ground truth
    let gt_knns: Vec<Vec<u32>> = read_output_bin("../data/dummy-gt.bin", queries.len());

    // Calculate recall
    let recall = get_knn_recall(&knns, &gt_knns);
    println!("Recall: {}", recall);
    println!("Total time: {}", start.elapsed().as_secs());
}

fn read_bin(path: &str, dim: usize) -> Vec<Vec<f32>> {
    println!("Reading Data: {}", path);

    let file = File::open(Path::new(path)).unwrap();
    let mut reader = BufReader::new(file);

    let mut n_points = [0u8; 4];
    reader.read_exact(&mut n_points).unwrap();
    let n_points = i32::from_le_bytes(n_points) as usize;

    println!("# of points: {}", n_points);

    let mut data = vec![vec![0f32; dim]; n_points];
    let mut buffer = vec![0f32; dim];
    for i in 0..n_points {
        reader
            .read_exact(bytemuck::cast_slice_mut(&mut buffer))
            .unwrap();
        data[i] = buffer.clone();
    }

    println!("Finish Reading Data");
    data
}

fn read_output_bin(path: &str, size: usize) -> Vec<Vec<u32>> {
    println!("Reading Data: {}", path);

    let file = File::open(Path::new(path)).unwrap();
    let mut reader = BufReader::new(file);

    let mut data = vec![vec![0u32; K_NEAREST_NEIGHBOURS as usize]; size];
    let mut buffer = vec![0u32; K_NEAREST_NEIGHBOURS as usize];
    for i in 0..size {
        reader
            .read_exact(bytemuck::cast_slice_mut(&mut buffer))
            .unwrap();
        data[i] = buffer.clone();
    }

    println!("Finish Reading Data");
    data
}

fn get_knn_recall(knns: &Vec<Vec<u32>>, gt_knns: &Vec<Vec<u32>>) -> f32 {
    let mut correct = 0;
    for i in 0..knns.len() {
        for j in 0..knns[0].len() {
            for k in 0..knns[0].len() {
                if knns[i][k] == gt_knns[i][j] {
                    correct += 1;
                    break;
                }
            }
        }
    }
    correct as f32 / knns.len() as f32 / knns[0].len() as f32
}

fn brute_force(nodes: &Vec<Vec<f32>>, queries: &Vec<Vec<f32>>) -> Vec<Vec<u32>> {
    let mut result = vec![vec![0u32; K_NEAREST_NEIGHBOURS as usize]; queries.len()];

    for i in 0..queries.len() {
        let query_type = queries[i][0] as u32;
        let v = queries[i][1] as i32;
        let l = queries[i][2];
        let r = queries[i][3];
        let vec = &queries[i][4..QUERY_DIMENSIONS as usize];

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
        if pq.len() < K_NEAREST_NEIGHBOURS as usize {
            println!("id: {}", i);
            println!("query type: {} v: {} l: {} r: {}", query_type, v, l, r);
            println!("K: {}", pq.len());
        }
        for j in 0..K_NEAREST_NEIGHBOURS as usize {
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
