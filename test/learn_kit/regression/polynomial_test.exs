defmodule LearnKit.Regression.PolynomialTest do
  use ExUnit.Case
  alias LearnKit.Regression.Polynomial

  setup_all do
    factors = [-3, -2, -1, -0.2, 1, 3]
    results = [0.9, 0.8, 0.4, 0.2, 0.1, 0]
    {:ok, predictor: Polynomial.new(factors, results), factors: factors, results: results}
  end

  test "create new polynomial predictor with empty data set" do
    assert Polynomial.new() == %Polynomial{}
  end

  test "create new polynomial predictor with data" do
    assert Polynomial.new([1, 2], [3, 4]) == %Polynomial{coefficients: [], degree: 2, factors: [1, 2], results: [3, 4]}
  end

  test "fit data set", state do
    %Polynomial{coefficients: coefficients, degree: 2, factors: factors, results: results } = state.predictor |> Polynomial.fit(degree: 2)

    assert coefficients == [0.2290655593570844, -0.16280041315555793, 0.027763965678671648]
    assert factors == state.factors
    assert results == state.results
  end

  test "fit data set with degree of 4", state do
    %Polynomial{coefficients: coefficients, degree: 4, factors: factors, results: results } = state.predictor |> Polynomial.fit(degree: 4)

    assert coefficients == [0.14805723970909512, -0.15811217698985996, 0.12329778502873823, 8.627221168971827e-4, -0.009963024223179073]
    assert factors == state.factors
    assert results == state.results
  end

  test "predict using the polynomial model of simple sample", state do
    {:ok, result} = state.predictor |> Polynomial.fit(degree: 2) |> Polynomial.predict(3)

    assert result == -0.009459989001544572
  end

  test "predict using the polynomial model of multiple samples", state do
    {:ok, result} = state.predictor |> Polynomial.fit(degree: 2) |> Polynomial.predict([3, 5])

    assert result == [-0.009459989001544572, 0.10916263554608596]
  end

  test "returns coefficient of determination R^2 of the prediction", state do
    predictor = state.predictor |> Polynomial.fit()

    assert {:ok, result} = predictor |> Polynomial.score()
    assert result == 0.9614116660464942
  end
end
