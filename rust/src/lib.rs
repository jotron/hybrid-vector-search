use bytemuck;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

pub const VECTOR_DIMENSIONS: usize = 100;
pub const NODE_DIMENSIONS: usize = 102;
pub const QUERY_DIMENSIONS: usize = 104;
pub const K_NEAREST_NEIGHBOURS: usize = 100;

pub fn run_with_solver(solve: fn(nodes: Vec<Vec<f32>>, queries: Vec<Vec<f32>>) -> Vec<Vec<u32>>) {
    let start = Instant::now();

    let args: Vec<String> = env::args().collect();
    let source_path = &args[1];
    let query_path = &args[2];
    let mut gt_path = "";
    if args.len() > 3 {
        gt_path = &args[3];
    }

    // Read nodes & queries
    let nodes = read_bin(source_path, NODE_DIMENSIONS);
    let queries = read_bin(query_path, QUERY_DIMENSIONS);
    let num_nodes = nodes.len();
    let num_queries = queries.len();

    // Calculate
    let knns: Vec<Vec<u32>> = solve(nodes, queries);
    // Stop Timing
    let duration = start.elapsed().as_millis();

    println!(
        "Total time for {} nodes and {} queries: {}ms",
        num_nodes, num_queries, duration
    );

    // Read ground truth if available
    if gt_path != "" {
        println!("Calculating recall...");
        let gt_knns: Vec<Vec<u32>> = read_output_bin(gt_path, num_queries);
        let recall = get_knn_recall(&knns, &gt_knns);
        println!("  recall = {}", recall);
    }
}

fn read_bin(path: &str, dim: usize) -> Vec<Vec<f32>> {
    //println!("Reading Data: {}", path);

    let file = File::open(Path::new(path)).unwrap();
    let mut reader = BufReader::new(file);

    let mut n_points = [0u8; 4];
    reader.read_exact(&mut n_points).unwrap();
    let n_points = i32::from_le_bytes(n_points) as usize;

    //println!("# of points: {}", n_points);

    let mut data = vec![vec![0f32; dim]; n_points];
    let mut buffer = vec![0f32; dim];
    for i in 0..n_points {
        reader
            .read_exact(bytemuck::cast_slice_mut(&mut buffer))
            .unwrap();
        data[i] = buffer.clone();
    }

    //println!("Finish Reading Data");
    data
}

fn read_output_bin(path: &str, size: usize) -> Vec<Vec<u32>> {
    //println!("Reading Data: {}", path);

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
