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

- [Q-learning model](https://github.com/Flora-Liao/DRL_final_MsPacman/blob/main/DRL_final_code/code/Q/data/ckpt_episode_8000.pt)
- [PPO model](https://github.com/Flora-Liao/DRL_final_MsPacman/blob/main/DRL_final_code/code/PPO/data/mspacman_249.pkl)


## Results

Our model achieves the following performance on :

Q-learning learning curve:
![total_reward_episode_8000](https://github.com/Flora-Liao/DRL_final_MsPacman/assets/92087054/7c72a9ea-a914-45af-adc2-0e1cf8aa11e2)

PPO learning curve:
![total_rewards_250](https://github.com/Flora-Liao/DRL_final_MsPacman/assets/92087054/1c618d58-5be3-46b1-919f-d5d737bf33f8)


| Model name         | Average Rewards  | 
| ------------------ |---------------- | 
| Random agent   |    221        |  
| Q-learning agent   |     326        |   
|PPO agent  |     967         |   



