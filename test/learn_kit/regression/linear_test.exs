defmodule LearnKit.Regression.LinearTest do
  use ExUnit.Case

  alias LearnKit.Regression.Linear

  setup_all do
    {:ok, predictor: Linear.new([1, 2, 3, 4], [3, 6, 10, 15])}
  end

  test "create new linear predictor with empty data set" do
    assert %Linear{factors: factors, results: results, coefficients: coefficients} = Linear.new

    assert factors == []
    assert results == []
    assert coefficients == []
  end

  test "create new linear predictor with data", state do
    assert %Linear{factors: factors, results: results, coefficients: coefficients} = state[:predictor]

    assert factors == [1, 2, 3, 4]
    assert results == [3, 6, 10, 15]
    assert coefficients == []
  end

  test "fit data set", state do
    %Linear{coefficients: coefficients} = state[:predictor] |> Linear.fit

    assert coefficients == [-1.5, 4.0]
  end

  test "fit data set with gradient descent", state do
    %Linear{coefficients: coefficients} = state[:predictor] |> Linear.fit([method: "gradient descent"])

    assert [-1.5, 4.0] = coefficients |> Enum.map(fn x -> Float.round(x, 2) end)
  end

  test "return prediction using the linear model", state do
    predictor = state[:predictor] |> Linear.fit

    assert {:ok, result} = predictor |> Linear.predict([4, 8, 13])
    assert result == [14.5, 30.5, 50.5]
  end

  test "returns coefficient of determination R^2 of the prediction", state do
    predictor = state[:predictor] |> Linear.fit

    assert {:ok, result} = predictor |> Linear.score
    assert result == 0.9876543209876543
  end
end
