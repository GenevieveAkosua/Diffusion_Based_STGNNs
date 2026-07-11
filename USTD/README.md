# Towards Unified Diffusion Models for Spatio-Temporal Predictive Learning

**Abstract:** 
Spatio-temporal graph learning is a fundamental problem in modern urban systems. Existing approaches tackle different tasks independently, tailoring their models to unique task characteristics. These methods, however, fall short of modeling intrinsic uncertainties in the spatio-temporal data. Meanwhile, their specialized designs misalign with the current research efforts toward unifying spatio-temporal graph learning solutions. In this paper, we propose to model these tasks in a unified probabilistic perspective, viewing them as predictions based on conditional information with shared dependencies. Based on this proposal, we introduce Unified Spatio-Temporal Diffusion Models (USTD) to address the tasks uniformly under the uncertainty-aware diffusion framework. USTD is holistically designed, comprising a shared spatio-temporal encoder and attention-based denoising decoders that are task-specific. The encoder, optimized by pre-training strategies, effectively captures conditional spatio-temporal patterns. The decoders, utilizing attention mechanisms, generate predictions by leveraging learned patterns. Opting for forecasting and kriging, the decoders are designed as Spatial Gated Attention (SGA) and Temporal Gated Attention (TGA) for each task, with different emphases on the spatial and temporal dimensions. Combining the advantages of deterministic encoders and probabilistic decoders, USTD achieves state-of-the-art performances compared to both deterministic and probabilistic baselines, while also providing valuable uncertainty estimates.

## Requirements
- [torch](https://pytorch.org/)
- numpy
- pandas

To install requirements (with out neptune):
```bash
pip install -r requirements.txt
```

## Datasets
* The processed datasets for our project is available at [Google Drive](https://drive.google.com/drive/folders/1hIx0GHBejBkIpxEAq61zyhUogPXFf2fo?usp=sharing).

Please put the processed data under [dataset](/data) folder.

## Train / Evaluate USTD
To train and evaluate models, please run the following command:
```bash
./train.sh [model] [dataset] [attribute] [time_length] [pretrain] [config] [batch] [gpu_ids] [seed]
```

| Task         | Dataset         | Command                                                                                     |
|--------------|-----------------|---------------------------------------------------------------------------------------------|
| Pre-train    | PEMS03          | ./train.sh gwavenet PEMS03 NA 12 NA config1 128 2 2030                                      |
| Pre-train    | PEMSBAY         | ./train.sh gwavenet PEMSBAY NA 12 NA config1 128 2 2030                                     |
| Pre-train    | BJAir           | ./train.sh gwavenet BJAir PM25 12 NA config1 128 2 2030                                     |
| Pre-train    | GZAir           | ./train.sh gwavenet GZAir PM25 12 NA config1 128 2 2030                                     |
| Forecasting  | PEMS03 | ./train.sh stdiffusionfore PEMS03 NA 24 **NAME_OF_PRETRAIN** config_PEMS03 64 3 2030        |
| Forecasting  | PEMSBAY | ./train.sh stdiffusionfore PEMSBAY NA 24 **NAME_OF_PRETRAIN** config_PEMSBAY 64 3 2030      |
| Forecasting  | BJAir | ./train.sh stdiffusionfore BJAir PM25 24 **NAME_OF_PRETRAIN** config_BJAir 64 1 2030        |
| Forecasting  | GZAir | ./train.sh stdiffusionfore GZAir PM25 24 **NAME_OF_PRETRAIN** config_GZAir 64 1 2030        |
| Kriging      | PEMS03 | ./train.sh stdiffusion PEMS03 NA 12 **NAME_OF_PRETRAIN** config_PEMS03 64 2 2030            |
| Kriging      | PEMSBAY | ./train.sh stdiffusion PEMSBAY NA 12 **NAME_OF_PRETRAIN** config_PEMSBAY 64 3 2030          |
| Kriging      | BJAir | ./train.sh stdiffusion BJAir PM25 12 **NAME_OF_PRETRAIN** config_BJAir 64 6 2030            |
| Kriging      | GZAir | ./train.sh stdiffusion GZAir PM25 12 **NAME_OF_PRETRAIN** config_GZAir 64 6 2030            |

Each running will train the model 3 times independently with the random seed increasing by 1. 
The framework will save the best model with the highest validation accuracy and evaluate it on the test set automatically after training. 
All the checkpoints and results will be saved at [checkpoints](checkpoints) folder.
For more training, testing, dataset configurations, please refer to [base_options](options/base_options.py), [train_options](options/train_options.py), [test_options](options/test_options.py), and [dataset_options](options/dataset_options.py).

## Reproduce Our Results
We saved the pretrained checkpoints for our USTD at [Google Drive](https://drive.google.com/drive/folders/1OCgxPe3gwWUjeOT5AWaIWqDS60MHUmhP?usp=sharing).
Download the checkpoints files and put them under [checkpoints](checkpoints) folder.
Each checkpoint file contains *run_test.sh* script. Please run the script to reproduce our results by the following command:
```bash
chmod u+x run_test.sh
./run_test.sh
```
The numerical results will be saved at *metrics.sh* and printed out.
The extrapolation results, ground truth, and the uncertainty estimates (if applicable) will be saved at *results.pkl*.
