# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram shows that the model is not perfectly calibrated. A perfectly calibrated model would have bucket accuracy close to the confidence level. In my results, several buckets show lower accuracy than their confidence level, which means the model is often over-confident.

For example, the 0.55 confidence bucket had an accuracy of 0.4641 with 362 examples. This means the model was around 55% confident, but it was correct only about 46% of the time. The 0.65 confidence bucket had an accuracy of 0.5395 with 354 examples, which also shows over-confidence.

The higher-confidence buckets performed better. The 0.85 bucket had an accuracy of 0.7877, and the 0.95 bucket had an accuracy of 0.8523. However, even these buckets are still slightly below their confidence level, so the model is still somewhat over-confident.

## Expected Calibration Error

The Expected Calibration Error for my model is:

ECE: 0.0829

This means that, on average, the model's confidence is about 0.0829 away from its actual accuracy across the confidence buckets. This is not extremely bad, but it shows that the model's confidence scores should not be fully trusted as production-level probabilities.

The model can still be useful for sentiment classification, but its confidence values should be treated carefully. A high confidence score does not always mean the model is correct.

## A specific calibration pattern

One specific pattern I noticed is over-confidence across the middle and high confidence buckets. The model often predicts with moderate or high confidence, but the actual accuracy is lower than the confidence level.

This may happen because the task has three sentiment classes: negative, neutral, and positive. Neutral examples are usually harder because they can be close to either positive or negative. The model may learn stronger patterns for clearly positive or clearly negative reviews, but it may struggle with ambiguous neutral reviews.

Because of this, the model can become confident even when the review is not clearly one class.

## A proposed engineering action

In production, I would add threshold-based abstention. If the model confidence is below 0.70, the system should not make a final automatic decision. Instead, it should mark the prediction for human review or return a message saying that the prediction is uncertain.

I would also use temperature scaling to improve calibration. This can adjust the model's confidence scores without fully retraining the model.

Finally, I would collect more training examples for confusing neutral cases. This would help the model learn better boundaries between neutral, positive, and negative reviews.