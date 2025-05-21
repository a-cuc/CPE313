
# Fatigue Detection using Hybrid Deep Learning Models

This project📜 component makes use of hybrid CNN-LSTM networks to detect fatigue😴 in videos, with the deep learning task being image classification between **fatigued** and **awake**.


## Dataset and Preprocessing

The dataset to be used would be the UTA-RLDD dataset containing RGB videos 🎥, which consist of 60 participants🧑‍🤝‍🧑 in three conditions: alert, low vigilant, and drowsy. However, the low vigilant videos are not used to simplify training.

Video frames are extracted and is grouped with the final size being:
```
 [batch size, sequence length, channels, height, width]
```

Data augmentation techniques such as Random Grayscaling and applying a Gaussian Blur to ensure model robustness

![image](https://github.com/user-attachments/assets/58b9c450-9e31-4c8f-9300-24e51ea3b605)



## Models

Three models are developed:
- Custom model

The entire group is passed through a `Conv3D` layer, applying `GroupNorm` and `weight_norm` to improve model training [1]. The `Conv3D` module in Pytorch also includes parameters such as *padding* and *groups* [2]. The output is then passed through a Stacked LSTM network.

- Transfer learning with EfficientNet

EfficientNet is used as the CNN backbone, with its output reshaped and fed into the Vanilla LSTM network.

- Transfer learning with ShuffleNet

ShuffleNet is used as the CNN backbone, with its output reshaped and fed into the Vanilla LSTM network.

Cosine Annealing with Warm Restarts is used as the scheduler to modify the learning rate per epochs [3]. The hyperparameters used for the models are as follows:
| Hyperparameter   | Value    | 
| :--------------- | :------- | 
| `epochs`         | 9        |
| `batch size`     | 8        |
| `learning rate`  | 1e-4     |
| `T_0 (scheduler)`| 3        |

References:

[1] https://arxiv.org/abs/1903.10520

[2] https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

[3] https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
## Metrics

As the task is classification, the main metric to be used is **accuracy** 🎯 that is compared between training and testing subsets.

## Testing

To test the model's accuracy, the best model is deployed thru **Streamlit**, where a video of an awake and fatigued person is inputted and the predicted class is compared to the actual class

Link: https://drive.google.com/file/d/1iJDZeUNcgjteuLaWyDYIh7TLc4EkzsSh/view?usp=drivesdk

Results show that the video correctly identified the awake and fatigued person
