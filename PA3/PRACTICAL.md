# Practical Infoirmation Answers

### 1. Reflect on the annotation task (PA3a). Briefly described how you experience the task. Suggests one or two way(s) to improve the the quality of labels.

There were a lot of human errors involved with spelling causing the dataset to be harder to handle. To fix this you could, instead of having the user type negative, positive or neutral manually have it set up so that the user can only select one of the three through a questionere or something similar. This removes the possibilty for differences in spelling or capitalization. To get better results you could also a bigger uneven amount of people label the same data to then take the average label answered as the final label. This reduces bias by the one performing the labeling.

### Explore the crowdsourced data.

#### - Are the labels as expected? If so, proceed to the next task. If not, describe the issue(s) found, fix the issue(s), and describe what you did.

In the crowdsourced data the issues mentioned above appeared again. There were several different spellings and capitalizations for the labels that caused problems when analising the data. To fix this we identified all unique labels, found the incorrectly inputted ones and mapped them to the "correct" version of the label.

#### - Check the distribution of the data. Describe what you observed.

The distribution of labels after the above mentioned cleaning was neutral: 47.5740%, positive: 30.1517% and negative: 22.2743%.

#### - Compare the crowdsourced data with the gold label data. Check the agreement score between the annotations in the two sets. We are going to assume that the crowdsourced data are done by one person and the gold annotations are done by one other person. Describe what you observed.

The agreement score (accuracy) between the two sets was approximately 65.5%. In terms of reliability, this is considered "Moderate" agreement. The confusion matrix revealed that the main source of disagreement was the 'neutral' class. The crowd annotators frequently mislabeled neutral tweets as either positive or negative, suggesting that distinguishing neutrality is the most subjective and difficult part of this annotation task.