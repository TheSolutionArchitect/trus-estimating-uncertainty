# How to run

- Update the config.json with input and output directory.
- The data should be 2D (.png) image of Transrectal Ultrasound.
- Create python environment with dependencies `Python 3.10` `torch=2.7.1+cu128`,`torchvision`, `torchaudio`, `monai=1.3.0`, `pandas`, `numpy`, `PIL`, `einops`
- run below command to execute the model run

```
python main.py --config config.json
```
