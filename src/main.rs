use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, Write},
    iter,
};

use tch::{Device, IndexOp, Kind, Tensor};

struct Coder {
    itoc: Vec<char>,
    ctoi: HashMap<char, i64>,
}

impl Coder {
    fn new(text: &str) -> Self {
        let mut itoc: Vec<char> = text
            .chars()
            .collect::<HashSet<_>>()
            .iter()
            .cloned()
            .collect();
        itoc.sort();

        let ctoi: HashMap<char, i64> = itoc
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i as i64))
            .collect();

        Self { itoc, ctoi }
    }

    fn to_i(&self, c: char) -> i64 {
        self.ctoi[&c]
    }

    fn to_c(&self, i: i64) -> char {
        self.itoc[i as usize]
    }
}

struct BigramDirect {
    device: Device,
    probabilities: Tensor,
}

impl BigramDirect {
    fn new(device: Device) -> Self {
        Self {
            device,
            probabilities: Tensor::ones(&[27, 27], (Kind::Int64, device)) / (27 * 27),
        }
    }

    fn train(&mut self, words: &[String], coder: &Coder) {
        let freqs = Tensor::ones(&[27, 27], (Kind::Int64, self.device));

        for words in words {
            for (a, b) in Self::word_to_bigrams(words) {
                let (i, j) = (coder.to_i(a), coder.to_i(b));

                let mut count = freqs.i((i, j));
                count += 1;
            }
        }

        let mut probabilities = (freqs + 1).to_kind(Kind::Float); // add smoothing

        probabilities /= probabilities.sum_dim_intlist([1].as_slice(), true, Kind::Float);

        self.probabilities = probabilities;
    }

    /// counts loss as negative log likelihood
    fn loss(&self, words: &[String], coder: &Coder) -> Tensor {
        let likelihoods = self.probabilities.log();

        let mut ll = Tensor::zeros(&[1], (Kind::Float, self.device));
        let mut n = Tensor::zeros(&[1], (Kind::Int64, self.device));

        for word in words {
            for (a, b) in Self::word_to_bigrams(word) {
                let (i, j) = (coder.to_i(a), coder.to_i(b));

                ll += likelihoods.i((i, j));
                n += 1;
            }
        }

        -ll / n
    }

    fn sample(&self, n_words: usize, coder: &Coder) -> Vec<String> {
        let mut words = vec![];
        let edge_index = coder.to_i('.');

        for _ in 0..n_words {
            let mut word = vec![];
            let mut idx = edge_index;

            loop {
                idx = self.probabilities.i(idx).multinomial(1, true).into();
                if idx == edge_index {
                    break;
                }

                word.push(coder.to_c(idx));
            }

            words.push(word.iter().collect::<String>());
        }

        words
    }

    fn word_to_bigrams(word: &str) -> Vec<(char, char)> {
        let chars = iter::once('.').chain(word.chars()).chain(iter::once('.'));

        chars.clone().zip(chars.skip(1)).collect()
    }
}

struct BigramNet {
    device: Device,
    weights: Tensor,
}

impl BigramNet {
    fn new(device: Device) -> Self {
        Self {
            device,
            weights: Tensor::randn(&[27, 27], (Kind::Float, device)).set_requires_grad(true),
        }
    }

    fn train(&mut self, words: &[String], n_steps: usize, coder: &Coder) {
        let (xs, ys) = self.dataset(words, coder);

        println!("Training on {} examples...", &xs.size()[0]);

        let ys = ys;

        for k in 0..n_steps {
            let loss = self.loss(&xs, &ys);
            let floss: f64 = (&loss).into();

            print!("Training step {}, loss: {}\r", k, floss);
            io::stdout().flush().expect("stdout should flush");

            self.weights.zero_grad();

            loss.backward();

            let mut weights_data = self.weights.data();
            weights_data += -50 * self.weights.grad();
        }
    }

    fn sample(&self, n_words: usize, coder: &Coder) -> Vec<String> {
        let mut words = vec![];
        let edge_index = coder.to_i('.');

        for _ in 0..n_words {
            let mut word = vec![];
            let mut idx = edge_index;

            loop {
                let encoded = Tensor::of_slice(&[idx]).to_device(self.device).one_hot(27);
                let probs = self.forward(&encoded);

                idx = probs.multinomial(1, true).into();

                if idx == edge_index {
                    break;
                }

                word.push(coder.to_c(idx));
            }

            words.push(word.iter().collect::<String>());
        }

        words
    }

    // -> probabilities
    fn forward(&self, encoded_xs: &Tensor) -> Tensor {
        // Linear layer
        let logits = encoded_xs.to_kind(Kind::Float).matmul(&self.weights); // => predicted log-counts

        // Softmax layer - eqiv. exp, then divide by row-wise sum
        logits.softmax(1, Kind::Float)
    }

    // Run a forward pass and calculate a loss
    fn loss(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
        let dataset_size = xs.size();
        let indexes = Tensor::arange(dataset_size[0], (Kind::Int64, self.device));

        let probs = self.forward(xs);

        let log_probs = -probs.index(&[Some(&indexes), Some(ys)]).log();
        assert_eq!(log_probs.size(), [dataset_size[0]]);

        -probs
            .index(&[Some(&indexes), Some(ys)])
            .log()
            .mean(Kind::Float)
    }

    fn dataset(&self, words: &[String], coder: &Coder) -> (Tensor, Tensor) {
        let (xs, ys): (Vec<i64>, Vec<i64>) = words
            .iter()
            .flat_map(|word| {
                Self::word_to_bigrams(word)
                    .iter()
                    .map(|(a, b)| (coder.to_i(*a), coder.to_i(*b)))
                    .collect::<Vec<_>>()
            })
            .unzip();

        (
            Tensor::of_slice(&xs)
                .to_device(self.device)
                .onehot(27)
                .to_kind(Kind::Float),
            Tensor::of_slice(&ys).to_device(self.device),
        )
    }

    // TODO return an interator
    fn word_to_bigrams(word: &str) -> Vec<(char, char)> {
        let chars = iter::once('.').chain(word.chars()).chain(iter::once('.'));

        chars.clone().zip(chars.skip(1)).collect()
    }
}

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
