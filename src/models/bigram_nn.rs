use std::{
    io::{self, Write},
    iter,
};
use tch::{Device, Kind, Tensor};

use crate::coder::Coder;

pub struct BigramNet {
    device: Device,
    weights: Tensor,
}

impl BigramNet {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            weights: Tensor::randn(&[27, 27], (Kind::Float, device)).set_requires_grad(true),
        }
    }

    pub fn train(&mut self, words: &[String], n_steps: usize, coder: &Coder) {
        let (xs, ys) = self.dataset(words, coder);

        println!("Training on {} examples...", &xs.size()[0]);

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

    pub fn sample(&self, n_words: usize, coder: &Coder) -> Vec<String> {
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
    pub fn loss(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
        let dataset_size = xs.size();
        let indexes = Tensor::arange(dataset_size[0], (Kind::Int64, self.device));

        let probs = self.forward(xs);

        let log_probs = probs.index(&[Some(&indexes), Some(ys)]).log();
        assert_eq!(log_probs.size(), [dataset_size[0]]);

        -log_probs.mean(Kind::Float)
    }

    pub fn dataset(&self, words: &[String], coder: &Coder) -> (Tensor, Tensor) {
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
