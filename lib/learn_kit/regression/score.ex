defmodule LearnKit.Regression.Score do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp calculate_score(%_{coefficients: []}, _, _), do: raise("There was no fit for model")

      defp calculate_score(regression = %_{coefficients: _, factors: _, results: results}) do
        1.0 - sum_of_squared_errors(regression) / total_sum_of_squares(results)
      end

      defp prediction_error(regression, x, y) do
        y - predict(regression, x)
      end

      defp sum_of_squared_errors(
             regression = %_{coefficients: _, factors: factors, results: results}
           ) do
        Enum.zip(factors, results)
        |> Enum.reduce(0, fn {xi, yi}, acc ->
          acc + squared_prediction_error(regression, xi, yi)
        end)
      end

      defp total_sum_of_squares(list) do
        mean_list = Math.mean(list)
        Enum.reduce(list, 0, fn x, acc -> acc + :math.pow(x - mean_list, 2) end)
      end

      defp squared_prediction_error(regression = %_{coefficients: coefficients}, x, y) do
        regression
        |> prediction_error(x, y)
        |> :math.pow(2)
      end
    end
  end
end
