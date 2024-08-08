# Large language models for machine learning interatomic potentials

This repo contains code for the paper 'Large language models for machine learning interatomic potentials: A user-centric approach to knowledge extraction from literature'

## Requirements and Setup

- Python 3.10
- Pytorch (version 1.12.1)
- Transformers (version 4.41.2)

You can install all required Python packages using the provided environment.yml file using `conda env create -f environment.yml`

## Running the code

The web application AnnoApp can be launched by the following command:
```bash
flask --app AnnoApp run
```

The script for the LLM fine-tuning can be run by using the following command:

```bash
python finetune.py --save_model True --seed 4232 --n_epochs 6
```

The MaterialsBERT language model that is used as the pre-trained model can be found [here](https://huggingface.co/pranav-s/MaterialsBERT)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE.md) file for details.
