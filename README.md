# NKRSI Workshop - Multi Level Perceptron

## Exercise

Fill in `train.py` with your code so there is MLP being trained on provided MNIST dataset.

Hint: Use `torch.nn.Linear` and `torch.nn.ReLU`

Try experimenting with different layer sizes and activation functions.


## Bonus

If using PyTorch was too easy, try implementing it using just `numpy` and formulas from the presentation.

Hints:
  - Weights between layers can be generated using `np.random.rand`
  - You can use MSE loss, i.e. `L(actual, predicted) = (actual - predicted)^2`
  - For activation you can use `ReLU(x) = min(0, x)`
  - Write `forward` function that performs forward propagation and returns `x` (before activation) and `z` (activation) for each layer
  - Write `backward` function that:
    - takes `x` and `z` at each layer from `forward` result
    - calculates loss and it's derivative
    - calculates `dy, df, dW, dz`
    - updates network weights `W -= learning_rate * dW`
  - Write `train` function that iterates `forward` and `backward` certain number of times