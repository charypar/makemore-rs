use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Lines},
    iter,
};

use tch::{Device, IndexOp, Kind, Tensor};

fn main() {
    let names: Vec<String> = read_names()
        .expect("should read names")
        .collect::<Result<Vec<_>, _>>()
        .expect("should read names");

    let (idx_to_char, char_to_idx) = make_mappings(&names.join("."));
    let edge_index = char_to_idx[&'.'];

    let bigram_frequencies = count_bigrams(&names, &char_to_idx);

    let mut bigram_probabilities = (bigram_frequencies + 1).to_kind(Kind::Float);
    bigram_probabilities /= bigram_probabilities.sum_dim_intlist([1].as_slice(), true, Kind::Float);

    let nll = neg_log_likelihood(&bigram_probabilities, &names, &char_to_idx);

    println!("Dataset negative log likelihood: {}", nll);

    for _ in 0..10 {
        let mut word = vec![];
        let mut idx = edge_index;

        loop {
            idx = bigram_probabilities.i(idx).multinomial(1, true).into();
            if idx == edge_index {
                break;
            }

            word.push(idx_to_char[idx as usize]);
        }

        println!("{}", word.iter().collect::<String>());
    }
}

fn neg_log_likelihood(
    bigram_probabilities: &Tensor,
    names: &[String],
    char_to_idx: &HashMap<char, i64>,
) -> f64 {
    let likelihoods = bigram_probabilities.log();

    let mut ll = Tensor::zeros(&[1], (Kind::Float, Device::Cpu));
    let mut n = Tensor::zeros(&[1], (Kind::Int64, Device::Cpu));

    for name in names {
        for (a, b) in word_to_bigrams(name) {
            let (i, j) = (char_to_idx[&a], char_to_idx[&b]);

            ll += likelihoods.i((i, j));
            n += 1;
        }
    }

    (-ll / n).into()
}

fn count_bigrams(names: &[String], char_to_idx: &HashMap<char, i64>) -> Tensor {
    let freqs = Tensor::zeros(&[27, 27], (Kind::Int64, Device::Cpu));

    for name in names {
        for (a, b) in word_to_bigrams(name) {
            let (i, j) = (char_to_idx[&a], char_to_idx[&b]);

            let mut count = freqs.i((i, j));
            count += 1;
        }
    }
    freqs
}

fn word_to_bigrams(word: &str) -> Vec<(char, char)> {
    let chars = iter::once('.').chain(word.chars()).chain(iter::once('.'));

    chars.clone().zip(chars.skip(1)).collect()
}

fn make_mappings(text: &str) -> (Vec<char>, HashMap<char, i64>) {
    let mut idx_to_char: Vec<char> = text
        .chars()
        .collect::<HashSet<_>>()
        .iter()
        .cloned()
        .collect();
    idx_to_char.sort();
    let char_to_idx: HashMap<char, i64> = idx_to_char
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i as i64))
        .collect();

    (idx_to_char, char_to_idx)
}

fn read_names() -> anyhow::Result<Lines<BufReader<File>>> {
    let names_file = File::open("names.txt")?;

    Ok(BufReader::new(names_file).lines())
}
