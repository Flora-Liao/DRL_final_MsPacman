# Training Q-learning and PPO Agents in the MsPacman-v5 Environment

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the Q-learning in the paper, run this command:

```train
python DRL_final_code/code/Q/Q_train.py
```

To train the PPO in the paper, run this command:

```train
python DRL_final_code/code/PPO/PPO_train.py
```


## Evaluation

To evaluate my model on ALE/MsPacman-v5, run:

```eval
# For Q-learning
python DRL_final_code/code/Q/Q_test.py

# For PPO
python DRL_final_code/code/PPO/PPO_test.py
```


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
