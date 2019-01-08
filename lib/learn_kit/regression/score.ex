defmodule LearnKit.Regression.Score do
  @moduledoc """
  Module for scoring regression models
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      @doc """
      Returns the coefficient of determination R^2 of the prediction

      ## Parameters

        - predictor: %LearnKit.Regression.Linear{}

      ## Examples

          iex> predictor |> LearnKit.Regression.Linear.score
          {:ok, 0.9876543209876543}

      """
      @spec score(%LearnKit.Regression.Linear{
              factors: factors,
              results: results,
              coefficients: coefficients
            }) :: {:ok, number}

      def score(regression = %_{factors: _, results: _, coefficients: _}) do
        {
          :ok,
          calculate_score(regression)
        }
      end

      defp calculate_score(%_{coefficients: []}, _, _), do: raise("There was no fit for model")

      defp calculate_score(regression = %_{coefficients: _, factors: _, results: results}) do
        1.0 - sum_of_squared_errors(regression) / total_sum_of_squares(results)
      end

      defp prediction_error(regression, x, y) do
        {:ok, prediction} = predict(regression, x)
        y - prediction
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
