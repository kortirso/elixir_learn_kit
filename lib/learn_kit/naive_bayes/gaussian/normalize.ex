defmodule LearnKit.NaiveBayes.Gaussian.Normalize do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Preprocessing

  defmacro __using__(_opts) do
    quote do
      defp normalize_data(data_set, type) do
        case type do
          t when t in ["minimax", "z_normalization"] -> normalize(data_set, type)
          _ -> data_set
        end
      end

      # normalize each feature
      defp normalize(data_set, type) do
        coefficients = find_coefficients_for_normalization(data_set, type)
        Enum.map(data_set, fn {key, features} ->
          {
            key,
            Enum.map(features, fn feature -> Preprocessing.normalize_feature(feature, coefficients, type) end)
          }
        end)
      end

      # find coefficients for normalization
      defp find_coefficients_for_normalization(data_set, type) do
        Enum.reduce(data_set, [], fn {_, features}, acc ->
          Enum.reduce(features, acc, fn feature, acc -> [feature | acc] end)
        end)
        |> Preprocessing.coefficients(type)
      end
    end
  end
end
