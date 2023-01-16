## Dialog Step by Step Learning
 
Dialogue Generation is the task of creating an appropriate response for a specific utterance. In the training session for this task, model learns how to generate the most appropriate answer based on utterance-response paired dataset. However trained dialogue generation model tends to frequently create general answers that would be appropriate in any situation. This is because the more common answers are, the less likely they are to be wrong, so the more common answers returned by the model trained with conditional probability, the better it is judged to have learned. To mend this problem, this repo presents and implements a methodology of training model with step-by-step training. The detailed learning strategy is as follows.


<br>

## Training Strategy

**First Step** <br>
 
Just like most sequence generation models do, our model learns to make whole prediction sequence.

<br>

**Second Step** <br>
After training with first strategy, the model then now learns to predict very first label token.
The loss in here is Generation Loss + First Token Prediction

<br>

**Third Step** <br>
Last, the model learns to make a prediction sequence with proper length. Often model predicts too long or short sequences.


<br><br>

## Configurations

<br><br>

## Results

<br><br>

## How to Use

<br><br>

## References

<br>
