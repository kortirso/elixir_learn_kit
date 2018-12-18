# LearnKit

Elixir package for machine learning

Available preprocessing methods:

- Normalization

Available algorithms for prediction:

- Linear Regression

Available algorithms for classification:

- K-Nearest Neighbours
- Gaussian Naive Bayes

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `learn_kit` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:learn_kit, "~> 0.1.5"}
  ]
end
```

### Normalization

Normalize data set with minimax normalization

```elixir
  alias LearnKit.Preprocessing
  Preprocessing.normalize([[1, 2], [3, 4], [5, 6]])
```

Or normalize data set with selected type

```elixir
  Preprocessing.normalize([[1, 2], [3, 4], [5, 6]], [type: "z_normalization"])
```
  options - array of options

Additionally you can prepare coefficients for normalization

```elixir
  Preprocessing.coefficients([[1, 2], [3, 4], [5, 6]], "minimax")
```
    type - method of normalization, one of the [minimax|z_normalization], required

And then normalize 1 feature with predefined coefficients

```elixir
  Preprocessing.normalize_feature([1, 2], [{1, 5}, {2, 6}], "minimax")
```
    type - method of normalization, one of the [minimax|z_normalization], required

### Linear Regression

Initialize predictor with data:

```elixir
  alias LearnKit.Regression.Linear
  predictor = Linear.new([1, 2, 3, 4], [3, 6, 10, 15])
```

Fit data set with least squares method:

```elixir
  predictor = predictor |> Linear.fit
```

Fit data set with gradient descent method:

```elixir
  predictor = predictor |> Linear.fit([method: "gradient descent"])
```

Predict using the linear model:

```elixir
  predictor |> Linear.predict([4, 8, 13])
```
    samples - array of variables, required

Returns the coefficient of determination R^2 of the prediction:

```elixir
  predictor |> Linear.score
```

### K-Nearest Neighbours classification

Initialize classifier with data set consists from labels and features:

```elixir
  alias LearnKit.Knn
  classifier =
    Knn.new
    |> Knn.add_train_data({:a1, [-1, -1]})
    |> Knn.add_train_data({:a1, [-2, -1]})
    |> Knn.add_train_data({:a2, [1, 1]})
```

Predict label for new feature:

```elixir
  Knn.classify(classifier, [feature: [-1, -2], k: 3, weight: "distance", normalization: "minimax"])
```
    feature - new feature for prediction, required
    k - number of nearest neighbors, optional, default - 3
    algorithm - algorithm for calculation of distances, one of the [brute], optional, default - "brute"
    weight - method of weighted neighbors, one of the [uniform|distance], optional, default - "uniform"
    normalization - method of normalization, one of the [none|minimax|z_normalization], optional, default - "none"

### Gaussian Naive Bayes classification

Initialize classifier with data set consists from labels and features:

```elixir
  alias LearnKit.NaiveBayes.Gaussian
  classifier =
    Gaussian.new
    |> Gaussian.add_train_data({:a1, [-1, -1]})
    |> Gaussian.add_train_data({:a1, [-2, -1]})
    |> Gaussian.add_train_data({:a2, [1, 1]})
```

Normalize data set:

```elixir
  classifier = classifier |> Gaussian.normalize_train_data("minimax")
```
    type - method of normalization, one of the [none|minimax|z_normalization], optional, default - "none"

Fit data set:

```elixir
  classifier = classifier |> Gaussian.fit
```

Return probability estimates for the feature:

```elixir
  classifier |> Gaussian.predict_proba([1, 2])
```
    feature - new feature for prediction, required

Return exact prediction for the feature:

```elixir
  classifier |> Gaussian.predict([1, 2])
```
    feature - new feature for prediction, required

Returns the mean accuracy on the given test data and labels:

```elixir
  classifier |> Gaussian.score
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/kortirso/elixir_learn_kit.

## License

The package is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

## Disclaimer

Use this package at your own peril and risk.

## Documentation

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/learn_kit](https://hexdocs.pm/learn_kit).

