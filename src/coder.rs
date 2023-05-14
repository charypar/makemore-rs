use std::collections::{HashMap, HashSet};

pub struct Coder {
    itoc: Vec<char>,
    ctoi: HashMap<char, i64>,
}

impl Coder {
    pub fn new(text: &str) -> Self {
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

    pub fn to_i(&self, c: char) -> i64 {
        self.ctoi[&c]
    }

    pub fn to_c(&self, i: i64) -> char {
        self.itoc[i as usize]
    }
}
