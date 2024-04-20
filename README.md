Trained a pre-trained resnet18, freezed the original weights, added a classifier and finetuned the classifier layers `Training_CelebA.ipynb`

Evaluated on the test data and obtained accuracies wrt every attribute `inference.py`

On the basis of some analysis including the data population and above results, chose "Male" as the spurious attribute abd "Bald" as the target attribute

Then trained a Binary Classisifer (a finetuned resnet18) and observed the bias (Equal Odds).
`Bald_Classifier(Biased).ipynb`, `equal_odds.py`.

Implemented an adversarial training pipeline to reduce the above bias.
`adversarial_training.ipynb`

Lastly, rechecked the bias again to ensure Equal Odds is satisfied.
`recheck_odds.py`

Conclusion: Adversarial Training did reduce the original bias by 20%.
