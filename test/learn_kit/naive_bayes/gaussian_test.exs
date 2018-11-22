defmodule LearnKit.NaiveBayes.GaussianTest do
  use ExUnit.Case

  alias LearnKit.NaiveBayes.Gaussian

  test "create new knn classificator with empty data set" do
    assert %Gaussian{data_set: data_set} = Gaussian.new

    assert data_set == []
  end

  test "add train data to classificator" do
    %Gaussian{data_set: data_set} = Gaussian.new
                    |> Gaussian.add_train_data({:a1, [1, 2]})
                    |> Gaussian.add_train_data({:a1, [1, 3]})
                    |> Gaussian.add_train_data({:b1, [2, 3]})

    assert data_set == [b1: [[2, 3]], a1: [[1, 3], [1, 2]]]
  end

  test "fit data set" do
    classificator = Gaussian.new([{:label1, [[-1, -1], [-2, -1], [-3, -2]]}, {:label2, [[1, 1], [2, 1], [3, 2], [-2, -2]]}])
    %Gaussian{fit_data: fit_data} = classificator |> Gaussian.fit

    assert fit_data == [
                        label1: [
                                  %{mean: -2.0, standard_deviation: 0.816496580927726, variance: 0.6666666666666666},
                                  %{mean: -1.3333333333333333, standard_deviation: 0.4714045207910317, variance: 0.2222222222222222}
                                ],
                        label2: [
                                  %{mean: 1.0, standard_deviation: 1.8708286933869707, variance: 3.5},
                                  %{mean: 0.5, standard_deviation: 1.5, variance: 2.25}
                                ]
                        ]
  end

  test "return probability estimates for the feature" do
    classificator = Gaussian.new([{:label1, [[-1, -1], [-2, -1], [-3, -2]]}, {:label2, [[1, 1], [2, 1], [3, 2], [-2, -2]]}])
    classificator = classificator |> Gaussian.fit

    assert {:ok, result} = classificator |> Gaussian.predict_proba([1, 2])
    assert result == [label1: 0.0, label2: 0.017199571]
  end

  test "return exact prediction for the feature" do
    classificator = Gaussian.new([{:label1, [[-1, -1], [-2, -1], [-3, -2]]}, {:label2, [[1, 1], [2, 1], [3, 2], [-2, -2]]}])
    classificator = classificator |> Gaussian.fit

    assert {:ok, result} = classificator |> Gaussian.predict([1, 2])
    assert result == {:label2, 0.017199571}
  end

  test "returns the mean accuracy on the given test data and labels" do
    classificator = Gaussian.new([{:label1, [[-1, -1], [-2, -1], [-3, -2]]}, {:label2, [[1, 1], [2, 1], [3, 2], [-2, -2]]}])
    classificator = classificator |> Gaussian.fit

    assert {:ok, result} = classificator |> Gaussian.score
    assert result == 0.857143
  end
end
