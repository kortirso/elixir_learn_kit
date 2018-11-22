defmodule LearnKit.Regression.Linear.Predict do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp predict_sample(sample, [alpha, beta]) do
        sample * beta + alpha
      end

      defp calculate_score([], _, _), do: raise("There was no fit for model")

      defp calculate_score(coefficients, factors, results) do
        1.0 - sum_of_squared_errors(coefficients, factors, results) / total_sum_of_squares(results)
      end

      defp total_sum_of_squares(list) do
        mean_list = Math.mean(list)
        list
        |> Enum.map(fn x -> :math.pow(x - mean_list, 2) end)
        |> Enum.sum
      end

      defp sum_of_squared_errors(coefficients, factors, results) do
        Enum.zip(factors, results)
        |> Enum.map(fn {xi, yi} -> :math.pow(prediction_error(coefficients, xi, yi), 2) end)
        |> Enum.sum
      end

      defp prediction_error(coefficients, x, y) do
        y - predict_sample(x, coefficients)
      end
    end
  end
end
