mod coder;
mod models;

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use tch::Device;

use coder::Coder;
use models::BigramDirect;
use models::BigramNet;

fn main() {
    let device = Device::Cpu;

    let names: Vec<String> =
        BufReader::new(File::open("names.txt").expect("Could not open names.txt"))
            .lines()
            .collect::<Result<Vec<_>, _>>()
            .expect("should read names");

    let coder = Coder::new(&names.join("."));

    // Make and train models

    let mut direct = BigramDirect::new(device);
    direct.train(&names, &coder);

    let mut neural_net = BigramNet::new(device);
    neural_net.train(&names, 300, &coder);

    // Print achieved losses

    let loss: f64 = direct.loss(&names, &coder).into();
    println!("Direct model loss: {}", loss);

    let (xs, ys) = neural_net.dataset(&names, &coder);
    let loss: f64 = neural_net.loss(&xs, &ys).into();
    println!("Neural net model loss: {}", loss);

    // Sample both models for comparison

    println!("\n------");
    println!("Samples from direct model:");
    for word in direct.sample(5, &coder) {
        println!("{}", word);
    }

    println!("\n------");
    println!("Samples from neural net:");
    for word in neural_net.sample(5, &coder) {
        println!("{}", word);
    }
}
