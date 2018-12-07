defmodule LearnKit.NaiveBayes.Gaussian.Score do
  @moduledoc """
  Module for calculating accuracy of prediction
  """

  alias LearnKit.NaiveBayes.Gaussian
  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp calc_score(fit_data, data_set) do
        data_set
        |> Enum.map(fn {label, features} ->
          check_features(features, fit_data, label)
        end)
        |> List.flatten()
        |> Math.mean()
        |> Float.ceil(6)
      end

      defp check_features(features, fit_data, label) do
        Enum.map(features, fn feature ->
          check_feature(feature, fit_data, label)
        end)
      end

      defp check_feature(feature, fit_data, label) do
        {:ok, {predicted_label, _}} = Gaussian.predict(%Gaussian{fit_data: fit_data}, feature)
        if predicted_label == label, do: 1, else: 0
      end
    end
  end
end
