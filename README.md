# LearnKit

Elixir package for machine learning

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `learn_kit` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:learn_kit, "~> 0.0.9"}
  ]
end
```

### K-Nearest Neighbours

Initialize classificator with data set consists from labels and features:

```elixir
  classificator = LearnKit.Knn.new
                  |> LearnKit.Knn.add_train_data({:a1, [-1, -1]})
                  |> LearnKit.Knn.add_train_data({:a1, [-2, -1]})
                  |> LearnKit.Knn.add_train_data({:a1, [-3, -2]})
                  |> LearnKit.Knn.add_train_data({:a2, [1, 1]})
                  |> LearnKit.Knn.add_train_data({:a2, [2, 1]})
                  |> LearnKit.Knn.add_train_data({:a2, [3, 2]})
                  |> LearnKit.Knn.add_train_data({:a2, [-2, -2]})
```

Predict label for new feature:

```elixir
  LearnKit.Knn.classify(classificator, [feature: [-1, -2], k: 3, weight: "distance"])
```
    feature - new feature for prediction, required
    k - number of nearest neighbors, optional, default - 3
    algorithm - algorithm for calculation of distances, one of the [brute], optional, default - "brute"
    weight - method of weighted neighbors, one of the [uniform|distance], optional, default - "uniform"

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

