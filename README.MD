# Large language models for machine learning interatomic potentials

This repo contains code for the paper 'Large language models for machine learning interatomic potentials: A user-centric approach to knowledge extraction from literature'

## Requirements and Setup

- Python 3.10
- Pytorch (version 1.10.0)
- Transformers (version 4.17.0)

You can install all required Python packages using the provided environment.yml file using `conda env create -f environment.yml`

## Running the code

The script for fine-tuning of the masked language model can be run by using the following command:

```bash
python run_mlm.py \
    --model_name_or_path bert-base \
    --train_file /path/to/train/file \
    --do_train \
    --do_eval \
    --output_dir /output
```

Use python data_extraction.py to combine NER predictions using heuristic rules.

The NER model used for sequence labeling can be found [here](https://huggingface.co/pranav-s/PolymerNER)

The MaterialsBERT language model that is used as the encoder for the above NER model can be found [here](https://huggingface.co/pranav-s/MaterialsBERT)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE.md) file for details.
