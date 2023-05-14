use std::iter;

use tch::{Device, IndexOp, Kind, Tensor};

use crate::coder::Coder;

pub struct BigramDirect {
    device: Device,
    probabilities: Tensor,
}

impl BigramDirect {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            probabilities: Tensor::ones(&[27, 27], (Kind::Int64, device)) / (27 * 27),
        }
    }

    pub fn train(&mut self, words: &[String], coder: &Coder) {
        let freqs = Tensor::ones(&[27, 27], (Kind::Int64, self.device));

        for words in words {
            for bigram in Self::word_to_bigrams(words) {
                let (i, j) = (coder.to_i(bigram.0), coder.to_i(bigram.1));

                let mut count = freqs.i((i, j));
                count += 1;
            }
        }

        let mut probabilities = (freqs + 1).to_kind(Kind::Float); // add smoothing

        probabilities /= probabilities.sum_dim_intlist([1].as_slice(), true, Kind::Float);

        self.probabilities = probabilities;
    }

    /// counts loss as negative log likelihood
    pub fn loss(&self, words: &[String], coder: &Coder) -> Tensor {
        let likelihoods = self.probabilities.log();

        let mut ll = Tensor::zeros(&[1], (Kind::Float, self.device));
        let mut n = Tensor::zeros(&[1], (Kind::Int64, self.device));

        for word in words {
            for bigram in Self::word_to_bigrams(word) {
                let (i, j) = (coder.to_i(bigram.0), coder.to_i(bigram.1));

                ll += likelihoods.i((i, j));
                n += 1;
            }
        }

        -ll / n
    }

    pub fn sample(&self, n_words: usize, coder: &Coder) -> Vec<String> {
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
