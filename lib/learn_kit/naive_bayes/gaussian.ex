defmodule LearnKit.NaiveBayes.Gaussian do
  @moduledoc """
  Module for Gaussian NB algorithm
  """

  defstruct data_set: []

  alias LearnKit.{NaiveBayes}

  @type label :: atom
  @type feature :: [integer]
  @type point :: {label, feature}
  @type features :: [feature]
  @type data_set :: [{label, features}]

  @doc """
  Creates classificator with empty data_set

  ## Examples

      iex> classificator = LearnKit.NaiveBayes.Gaussian.new
      %LearnKit.NaiveBayes.Gaussian{data_set: []}

  """
  @spec new() :: %LearnKit.NaiveBayes.Gaussian{data_set: []}

  def new do
    []
    |> NaiveBayes.Gaussian.new
  end

  @doc """
  Creates classificator with data_set

  ## Parameters

    - data_set: Keyword list with labels and features in tuples

  ## Examples

      iex> classificator = LearnKit.NaiveBayes.Gaussian.new([{:a1, [[1, 2], [2, 3]]}, {:b1, [[-1, -2]]}])
      %LearnKit.NaiveBayes.Gaussian{data_set: [a1: [[1, 2], [2, 3]], b1: [[-1, -2]]]}

  """
  @spec new(data_set) :: %LearnKit.NaiveBayes.Gaussian{data_set: data_set}

  def new(data_set) do
    %NaiveBayes.Gaussian{data_set: data_set}
  end

  @doc """
  Add train data to classificator

  ## Parameters

    - classificator: %LearnKit.NaiveBayes.Gaussian{}
    - train data: tuple with label and feature

  ## Examples

      iex> classificator |> LearnKit.NaiveBayes.Gaussian.add_train_data({:a1, [-1, -1]})
      %LearnKit.NaiveBayes.Gaussian{data_set: [a1: [[-1, -1]]]}

  """
  @spec add_train_data(%LearnKit.NaiveBayes.Gaussian{data_set: data_set}, point) :: %LearnKit.NaiveBayes.Gaussian{data_set: data_set}

  def add_train_data(%NaiveBayes.Gaussian{data_set: data_set}, {key, value}) do
    features = if Keyword.has_key?(data_set, key), do: Keyword.get(data_set, key), else: []
    data_set = Keyword.put(data_set, key, [value | features])
    %NaiveBayes.Gaussian{data_set: data_set}
  end
end
