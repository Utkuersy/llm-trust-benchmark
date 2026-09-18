# ML Model Lifecycle Standard

Every model deployed to production requires a Model Card. A Model Card
must contain at least the following sections: purpose, training data,
metrics, known limitations, and ethical review.

Production models are re-evaluated every 3 months. If performance drops
more than 5 points below baseline, the model is withdrawn.

The data drift monitoring threshold, measured by Population Stability
Index (PSI), is set at 0.20.
