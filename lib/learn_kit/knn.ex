defmodule LearnKit.Knn do
  @moduledoc """
  Module for k-nearest neighbours (knn) algorithm
  """

  defstruct data_set: []

  alias LearnKit.{Knn}

  use Knn.Classify

  @type label :: atom
  @type feature :: [integer]
  @type point :: {label, feature}
  @type features :: [feature]
  @type data_set :: [{label, features}]

  @doc """
  Creates classificator with empty data_set

  ## Examples

      iex> classificator = LearnKit.Knn.new
      %LearnKit.Knn{data_set: []}

  """
  @spec new() :: %LearnKit.Knn{data_set: []}

  def new do
    []
    |> Knn.new
  end

  @doc """
  Creates classificator with data_set

  ## Parameters

    - data_set: Keyword list with labels and features in tuples

  ## Examples

      iex> classificator = LearnKit.Knn.new([{:a1, [[1, 2], [2, 3]]}, {:b1, [[-1, -2]]}])
      %LearnKit.Knn{data_set: [a1: [[1, 2], [2, 3]], b1: [[-1, -2]]]}

  """
  @spec new(data_set) :: %LearnKit.Knn{data_set: data_set}

  def new(data_set) do
    %Knn{data_set: data_set}
  end

  @doc """
  Add train data to classificator

  ## Parameters

    - classificator: %LearnKit.Knn{}
    - train data: tuple with label and feature

  ## Examples

      iex> classificator |> LearnKit.Knn.add_train_data({:a1, [-1, -1]})
      %LearnKit.Knn{data_set: [a1: [[-1, -1]]]}

  """
  @spec add_train_data(%LearnKit.Knn{data_set: data_set}, point) :: %LearnKit.Knn{data_set: data_set}

  def add_train_data(%Knn{data_set: data_set}, {key, value}) do
    features = if Keyword.has_key?(data_set, key), do: Keyword.get(data_set, key), else: []
    data_set = Keyword.put(data_set, key, [value | features])
    %Knn{data_set: data_set}
  end

  @doc """
  Classify label of the new feature

  ## Parameters

    - classificator: %LearnKit.Knn{}
    - options: keyword list with options

  ## Options

    - feature: feature for classification, required, example: [1, 2, 3]
    - k: number of nearest neighbours, default is 3, optional
    - algorithm: brute, optional
    - weight: uniform/distance, default is uniform, optional

  ## Examples

      iex> classificator |> LearnKit.Knn.classify([feature: [-1, -2], k: 3, weight: "distance"])
      {:ok, "Feature has label a1 with weight 1.5"}

  """
  @spec classify(%LearnKit.Knn{data_set: data_set}, [tuple]) :: {atom, String.t}

  def classify(%Knn{data_set: data_set}, options \\ []) do
    try do
      unless Keyword.has_key?(options, :feature) do
        raise "Feature option is required"
      end
      # modification of options
      options = Keyword.merge([k: 3, algorithm: "brute", weight: "uniform"], options)
      # prediction
      {label, weight} = prediction(data_set, options)
      {:ok, "Feature has label #{label} with weight #{weight}"}
    rescue
      error -> {:error, error.message}
    end
  end
end
