defmodule LearnKit.Preprocessing do
  @moduledoc """
  Module for data preprocessing
  """

  alias LearnKit.{Preprocessing, Math}

  use Preprocessing.Normalize

  @type row :: [number]
  @type matrix :: [row]

  @doc """
  Normalize data set with minimax normalization

  ## Parameters

    - features: list of features for normalization

  ## Examples

      iex> LearnKit.Preprocessing.normalize([[1, 2], [3, 4], [5, 6]])
      [
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
      ]

  """
  @spec normalize(matrix) :: matrix

  def normalize(features) when is_list(features), do: normalize(features, [type: "minimax"])

  @doc """
  Normalize data set

  ## Parameters

    - features: list of features for normalization
    - options: keyword list with options

  ## Options

    - type: minimax/z_normalization, default is minimax, optional

  ## Examples

      iex> LearnKit.Preprocessing.normalize([[1, 2], [3, 4], [5, 6]], [type: "z_normalization"])
      [
        [-1.224744871391589, -1.224744871391589],
        [0.0, 0.0],
        [1.224744871391589, 1.224744871391589]
      ]

  """
  @spec normalize(matrix, list) :: matrix

  def normalize(features, options) when is_list(features) and is_list(options) do
    options = Keyword.merge([type: "minimax"], options)
    case options[:type] do
      "z_normalization" -> normalization(features, "z_normalization")
      _ -> normalization(features, "minimax")
    end
  end

  @doc """
  Prepare coefficients for normalization

  ## Parameters

    - features: features grouped by index
    - type: minimax/z_normalization

  ## Examples

      iex> LearnKit.Preprocessing.coefficients([[1, 2], [3, 4], [5, 6]], "minimax")
      [{1, 5}, {2, 6}]

      iex> LearnKit.Preprocessing.coefficients([[1, 2], [3, 4], [5, 6]], "z_normalization")
      [{3.0, 1.632993161855452}, {4.0, 1.632993161855452}]

  """
  @spec coefficients(matrix, String.t()) :: matrix

  def coefficients(features, type) when is_list(features) and is_binary(type) do
    features
    |> Math.transpose()
    |> Enum.map(fn list -> return_params(list, type) end)
  end

  @doc """
  Normalize 1 feature with predefined coefficients

  ## Parameters

    - feature: feature for normalization
    - coefficients: predefined coefficients
    - type: minimax/z_normalization

  ## Examples

      iex> LearnKit.Preprocessing.normalize_feature([1, 2], [{1, 5}, {2, 6}], "minimax")
      [0.0, 0.0]

  """
  @spec normalize_feature(list, list(tuple), String.t()) :: list

  def normalize_feature(feature, coefficients, type) when is_list(feature) and is_list(coefficients) and is_binary(type) do
    Enum.zip(feature, coefficients)
    |> Enum.map(fn {point, params_for_point} ->
      divider = define_divider(params_for_point, type)
      case divider do
        0 -> point
        _ -> (point - elem(params_for_point, 0)) / divider
      end
    end)
  end
end
