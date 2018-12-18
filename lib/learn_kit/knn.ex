defmodule LearnKit.Knn do
  @moduledoc """
  Module for k-nearest neighbours (knn) algorithm
  """

  defstruct data_set: []

  alias LearnKit.Knn

  use Knn.Classify

  @type label :: atom
  @type feature :: [integer]
  @type point :: {label, feature}
  @type features :: [feature]
  @type data_set :: [{label, features}]

  @doc """
  Creates classifier with empty data_set

  ## Examples

      iex> classifier = LearnKit.Knn.new
      %LearnKit.Knn{data_set: []}

  """
  @spec new() :: %Knn{data_set: []}

  def new do
    Knn.new([])
  end

  @doc """
  Creates classifier with data_set

  ## Parameters

    - data_set: Keyword list with labels and features in tuples

  ## Examples

      iex> classifier = LearnKit.Knn.new([{:a1, [[1, 2], [2, 3]]}, {:b1, [[-1, -2]]}])
      %LearnKit.Knn{data_set: [a1: [[1, 2], [2, 3]], b1: [[-1, -2]]]}

  """
  @spec new(data_set) :: %Knn{data_set: data_set}

  def new(data_set) when is_list(data_set) do
    %Knn{data_set: data_set}
  end

  @doc """
  Add train data to classifier

  ## Parameters

    - classifier: %LearnKit.Knn{}
    - train data: tuple with label and feature

  ## Examples

      iex> classifier = classifier |> LearnKit.Knn.add_train_data({:a1, [-1, -1]})
      %LearnKit.Knn{data_set: [a1: [[-1, -1]]]}

  """
  @spec add_train_data(%Knn{data_set: data_set}, point) :: %Knn{data_set: data_set}

  def add_train_data(%Knn{data_set: data_set}, {key, value}) when is_atom(key) and is_list(value) do
    features = if Keyword.has_key?(data_set, key), do: Keyword.get(data_set, key), else: []
    data_set = Keyword.put(data_set, key, [value | features])
    %Knn{data_set: data_set}
  end

  @doc """
  Classify label of the new feature

  ## Parameters

    - classifier: %LearnKit.Knn{}
    - options: keyword list with options

  ## Options

    - feature: feature for classification, required, example: [1, 2, 3]
    - k: number of nearest neighbours, default is 3, optional
    - algorithm: brute, optional
    - weight: uniform/distance, default is uniform, optional
    - normalization: none/minimax/z_normalization, default is none, optional

  ## Examples

      iex> classifier |> LearnKit.Knn.classify([feature: [-1, -2], k: 3, weight: "distance"])
      {:ok, :a1}

  """
  @spec classify(%Knn{data_set: data_set}, [tuple]) :: {:ok, label}

  def classify(%Knn{data_set: data_set}, options) when is_list(options) do
    cond do
      !Keyword.has_key?(options, :feature) ->
        {:error, "Feature option is required"}

      !is_list(Keyword.get(options, :feature)) ->
        {:error, "Feature option must be presented as array"}

      Keyword.has_key?(options, :k) && (!is_integer(Keyword.get(options, :k)) || Keyword.get(options, :k) <= 0) ->
        {:error, "K option must be positive integer"}

      true ->
        options = Keyword.merge([k: 3, algorithm: "brute", weight: "uniform", normalization: "none"], options)
        {label, _} = prediction(data_set, options)
        {:ok, label}
    end
  end
end
