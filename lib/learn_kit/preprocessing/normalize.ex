defmodule LearnKit.Preprocessing.Normalize do
  @moduledoc """
  Module for data normalization
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp minimax_normalization(features) do
        features_by_index = Math.transpose(features)
        list_of_params =
          features_by_index
          |> Enum.map(fn list ->
            {
              Enum.min(list),
              Enum.max(list)
            }
          end)
        features_by_index
        |> Enum.with_index()
        |> Enum.map(fn {feature, index} ->
          params_for_point = Enum.at(list_of_params, index)
          range = elem(params_for_point, 1) - elem(params_for_point, 0)
          feature
          |> Enum.map(fn point ->
            (point - elem(params_for_point, 0)) / range
          end)
        end)
        |> Math.transpose()
      end

      defp z_normalization(features) do
        features_by_index = Math.transpose(features)
        list_of_params =
          features_by_index
          |> Enum.map(fn list ->
            {
              Math.mean(list),
              Math.standard_deviation(list)
            }
          end)
        features_by_index
        |> Enum.with_index()
        |> Enum.map(fn {feature, index} ->
          params_for_point = Enum.at(list_of_params, index)
          feature
          |> Enum.map(fn point ->
            (point - elem(params_for_point, 0)) / elem(params_for_point, 1)
          end)
        end)
        |> Math.transpose()
      end
    end
  end
end
