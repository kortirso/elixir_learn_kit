defmodule LearnKit.NaiveBayes.Gaussian.Fit do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp fit_data(data_set) do
        Enum.map(data_set, fn {key, value} ->
          {key, calc_features(value)}
        end)
      end

      defp calc_features(features) do
        features
        |> Math.transpose()
        |> calc_combination()
      end

      defp calc_combination(combinations) do
        Enum.map(combinations, fn combination ->
          mean = Math.mean(combination)
          variance = Math.variance(combination, mean)
          standard_deviation = Math.standard_deviation_from_variance(variance)
          %{mean: mean, variance: variance, standard_deviation: standard_deviation}
        end)
      end
    end
  end
end
