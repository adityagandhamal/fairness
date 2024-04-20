Trained a pre-trained resnet18, with its weights freezen, added a classifier, and finetuned the classifier layers `Training_CelebA.ipynb`

Evaluated on the test data and obtained accuracies wrt every attribute `inference.py`

Based on some analysis including on data population and above results, chose "Male" as the spurious attribute and "Bald" as the target(affecting) attribute.

Then trained a Binary Classifier (a finetuned resnet18) and observed the bias (Equalized Odds).
`Bald_Classifier(Biased).ipynb`, `equal_odds.py`.

Implemented an adversarial training pipeline to reduce the above bias.
`adversarial_training.ipynb`

Lastly, rechecked the bias to ensure Equalized Odds is satisfied.
`recheck_odds.py`

Conclusion: Adversarial Training did reduce the original bias by 20%.
