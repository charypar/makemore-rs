# Makemore, but in Rust

An experiment with language modeling, following Andrej Karpathy [Youtube Lectures](https://www.youtube.com/watch?v=PaCmpygFfXo)

Original Python implementation available [here](https://github.com/karpathy/makemore)

## Running instructions

Under covers, this uses pytorch, loaded dynamically. In order for the program to know where to find the library, you need to first

```sh
source set_pytorch_path.sh
```

This [sounds](https://github.com/LaurentMazare/tch-rs/issues/629) [temporary](https://github.com/pytorch/pytorch/issues/96046), but I couldn't (be bothered to) get it working without the step.
