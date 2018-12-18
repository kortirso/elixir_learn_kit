defmodule LearnKit.Preprocessing.Normalize do
  @moduledoc """
  Module for data normalization
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp normalization(features, type) do
        features_by_index = Math.transpose(features)
        list_of_params = Enum.map(features_by_index, fn list -> return_params(list, type) end)
        features_by_index
        |> Enum.with_index()
        |> Enum.map(fn {feature, index} -> transform_feature(feature, Enum.at(list_of_params, index), type) end)
        |> Math.transpose()
      end

      defp return_params(list, "minimax") do
        {
          Enum.min(list),
          Enum.max(list)
        }
      end

      defp return_params(list, "z_normalization") do
        {
          Math.mean(list),
          Math.standard_deviation(list)
        }
      end

      defp transform_feature(feature, params_for_point, type) do
        divider = define_divider(params_for_point, type)
        case divider do
          0 -> feature
          _ -> Enum.map(feature, fn point -> (point - elem(params_for_point, 0)) / divider end)
        end
      end

      defp define_divider(params_for_point, "minimax") do
        elem(params_for_point, 1) - elem(params_for_point, 0)
      end

      defp define_divider(params_for_point, "z_normalization") do
        elem(params_for_point, 1)
      end
    end
  end
end
