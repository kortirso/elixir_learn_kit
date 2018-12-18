defmodule LearnKit.Preprocessing do
  @moduledoc """
  Module for data preprocessing
  """

  alias LearnKit.Preprocessing

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

  def normalize(features) when is_list(features) do
    normalize(features, [type: "minimax"])
  end

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
    case Keyword.get(options, :type) do
      "z_normalization" -> normalization(features, "z_normalization")
      _ -> normalization(features, "minimax")
    end
  end
end
