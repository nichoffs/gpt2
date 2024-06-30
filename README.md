`full/` contains the training script for the full-scale pre-training run on the fineweb dataset, as well as the script to download the dataset itself.

`local/` contains a script for local training on the tiny-shakespeare dataset, which allows for easier local testing.

`local/` seems to be done. There are a few more things that need to get done before the full training run.

- [ ] Gradient accumulation
- [ ] Distributed Data Parallel
- [ ] FineWeb
- [ ] HellaSwag Eval
- [ ] Validation Loss
