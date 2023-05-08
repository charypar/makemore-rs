use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Lines},
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
    probabilities: Tensor,
}

impl BigramDirect {
    fn new() -> Self {
        Self {
            probabilities: Tensor::ones(&[27, 27], (Kind::Int64, Device::Cpu)) / (27 * 27),
        }
    }

    fn train(&mut self, words: &[String], coder: &Coder) {
        let freqs = Tensor::ones(&[27, 27], (Kind::Int64, Device::Cpu));

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
    fn loss(&self, words: &[String], coder: &Coder) -> f64 {
        let likelihoods = self.probabilities.log();

        let mut ll = Tensor::zeros(&[1], (Kind::Float, Device::Cpu));
        let mut n = Tensor::zeros(&[1], (Kind::Int64, Device::Cpu));

        for word in words {
            for (a, b) in Self::word_to_bigrams(word) {
                let (i, j) = (coder.to_i(a), coder.to_i(b));

                ll += likelihoods.i((i, j));
                n += 1;
            }
        }

        (-ll / n).into()
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

fn main() {
    let names: Vec<String> = read_names()
        .expect("should read names")
        .collect::<Result<Vec<_>, _>>()
        .expect("should read names");

    let coder = Coder::new(&names.join("."));
    let mut model = BigramDirect::new();

    model.train(&names, &coder);

    let nll = model.loss(&names, &coder);

    println!("Dataset negative log likelihood: {}", nll);

    for word in model.sample(10, &coder) {
        println!("{}", word);
    }
}

fn read_names() -> anyhow::Result<Lines<BufReader<File>>> {
    let names_file = File::open("names.txt")?;

    Ok(BufReader::new(names_file).lines())
}
