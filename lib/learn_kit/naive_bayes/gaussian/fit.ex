defmodule LearnKit.NaiveBayes.Gaussian.Fit do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp fit_data(data_set) do
        data_set
        |> Enum.map(fn {key, value} ->
          {key, calc_features(value)}
        end)
      end

      defp calc_features(features) do
        features
        |> Math.transpose
        |> Enum.map(fn feature ->
          mean = Math.mean(feature)
          variance = Math.variance(feature, mean)
          standard_deviation = Math.standard_deviation_from_variance(variance)
          %{mean: mean, variance: variance, standard_deviation: standard_deviation}
        end)
      end
    end
  end
end
