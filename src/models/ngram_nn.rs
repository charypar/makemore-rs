use std::{
    io::{self, Write},
    iter,
};
use tch::{Device, Kind, Tensor};

use crate::coder::Coder;

pub struct NgramNet<const N: usize, const E: usize, const H: usize> {
    coder: Coder,
    emb_lookup: Tensor, // 27 by 2
    // Hidden layer
    w1: Tensor,
    b1: Tensor,
    // Top layer
    w2: Tensor,
    b2: Tensor,
}

impl<const N: usize, const E: usize, const H: usize> NgramNet<N, E, H> {
    pub fn new(device: Device, coder: Coder) -> Self {
        let emb_lookup =
            Tensor::randn(&[27, E as i64], (Kind::Float, device)).set_requires_grad(true);

        let w1 = Tensor::randn(&[(N * E) as i64, H as i64], (Kind::Float, device))
            .set_requires_grad(true);
        let b1 = Tensor::randn(&[H as i64], (Kind::Float, device)).set_requires_grad(true);

        let w2 = Tensor::randn(&[H as i64, 27], (Kind::Float, device)).set_requires_grad(true);
        let b2 = Tensor::randn(&[27], (Kind::Float, device)).set_requires_grad(true);

        Self {
            coder,
            emb_lookup,
            w1,
            b1,
            w2,
            b2,
        }
    }

    pub fn train(&mut self, xs: &Tensor, ys: &Tensor, n_steps: usize) {
        assert_eq!(xs.size().len(), 2);
        assert_eq!(xs.size()[1], N as i64);
        assert_eq!(xs.size()[0], ys.size()[0]);

        let rate = 0.1;

        println!("Training on {} examples...", &xs.size()[0]);

        for k in 0..n_steps {
            let loss = self.loss(xs, ys);

            let floss: f64 = (&loss).into();
            print!("Training step {}, loss: {}\r", k, floss);
            io::stdout().flush().expect("stdout should flush");

            self.emb_lookup.zero_grad();
            self.w1.zero_grad();
            self.b1.zero_grad();
            self.w2.zero_grad();
            self.b2.zero_grad();

            loss.backward();

            for p in [&self.emb_lookup, &self.w1, &self.b2, &self.w2, &self.b2] {
                let mut data = p.data();
                data += -rate * p.grad();
            }
        }
    }

    pub fn sample(&self, n_words: usize) -> Vec<String> {
        let mut words = vec![];
        let start = [0; N];
        let stop = 0;

        for _ in 0..n_words {
            let mut word = vec![];
            let mut input = start;

            loop {
                let encoded = Tensor::of_slice(&input).reshape(&[1, N as i64]);
                let probs = self.forward(&encoded);

                assert_eq!(probs.size(), [1, 27]);
                let output: i64 = probs.multinomial(1, true).into();

                if output == stop {
                    break;
                }

                word.push(self.coder.to_c(output));

                input.rotate_left(1);
                input[N - 1] = output;
            }

            words.push(word.iter().collect::<String>());
        }

        words
    }

    // -> probabilities
    fn forward(&self, xs: &Tensor) -> Tensor {
        // We choose to interpret the output as log(count), and because
        // we train the network with that interpretation, it will become so
        let logits = self.logits(xs);

        // Softmax layer - eqiv. exp, then divide by row-wise sum
        // This normalises the rows to be probability distribution of next letter
        logits.softmax(1, Kind::Float)
    }

    // Calculate a loss against known ys
    // xs is size-by-N-by-E dataset
    pub fn loss(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
        assert_eq!(xs.size().len(), 2);
        assert_eq!(xs.size()[1], N as i64);

        // cross entropy below is equivalent to
        // logits -> softmax (= exp -> normalise row-wise) -> select probs for Ys -> log -> mean
        //
        // in code:
        //
        // let dataset_size = xs.size();
        // let indexes = Tensor::arange(dataset_size[0], (Kind::Int64, self.device)); // => 0..N
        //
        // let log_probs = xs // one-hot encoded
        //     .to_kind(Kind::Float)
        //     .matmul(&self.weights) // => logits
        //     .softmax(1, Kind::Float) // => probs
        //     .log() // log-probs
        //     .index(&[Some(&indexes), Some(ys)]); // selected for targets
        //
        // assert_eq!(log_probs.size(), [dataset_size[0]]); // sanity check
        // -log_probs.mean(Kind::Float); // is negative log-likelihood

        // cross_entropy_for_logits is log_softmax, then nll_loss
        // does the same as above
        self.logits(xs).cross_entropy_for_logits(ys)
    }

    // Applies the linear layer of the network (just matrix multiplication)
    // This is the core of the model.
    //
    // => predicted log-counts ~ "logits"
    fn logits(&self, xs: &Tensor) -> Tensor {
        let n = xs.size()[0];
        assert_eq!(xs.size(), [n, N as i64]);

        let emb = self.embeddings(xs).view((-1, (E * N) as i64));
        assert_eq!(emb.size(), [n, (E * N) as i64]);

        let h = (emb.matmul(&self.w1) + &self.b1).tanh();
        assert_eq!(h.size(), [n, H as i64]);

        let logits = h.matmul(&self.w2) + &self.b2;
        assert_eq!(logits.size(), [n, 27]);

        logits
    }

    // Prepare a dataset based on the known words
    // Data set will be (n by N, n by 1) for n examples
    // Each row of x is N characters of input
    // Each row of y is expected output character
    pub fn dataset(&self, words: &[String]) -> (Tensor, Tensor) {
        let mut xs: Vec<[i64; N]> = vec![];
        let mut ys: Vec<i64> = vec![];

        for word in words {
            let mut context = [0; N];

            for ch in word.chars().chain(iter::once('.')) {
                let idx = self.coder.to_i(ch);

                xs.push(context);
                ys.push(idx);

                context.rotate_left(1);
                context[N - 1] = idx;
            }
        }

        let n = xs.len() as i64;
        let xs = xs.as_slice().concat();

        let x = Tensor::of_slice(&xs).view((n, N as i64));
        let y = Tensor::of_slice(&ys);

        (x, y)
    }

    /// Convert characters (their indexes) into a 2D space
    /// Through a trained lookup table
    pub fn embeddings(&self, xs: &Tensor) -> Tensor {
        let result = self.emb_lookup.index(&[Some(xs)]);

        let len = xs.size()[0];
        assert_eq!(result.size(), [len, N as i64, E as i64]);

        result
    }
}
