defmodule LearnKit.NaiveBayes.Gaussian do
  @moduledoc """
  Module for Gaussian NB algorithm
  """

  defstruct data_set: [], fit_data: []

  alias LearnKit.NaiveBayes.Gaussian

  use Gaussian.Fit
  use Gaussian.Classify
  use Gaussian.Score

  @type label :: atom
  @type feature :: [integer]
  @type prediction :: {label, number}
  @type predictions :: [prediction]
  @type point :: {label, feature}
  @type features :: [feature]
  @type data_set :: [{label, features}]
  @type fit_feature :: %{mean: float, standard_deviation: float, variance: float}
  @type fit_features :: [fit_feature]
  @type fit_data :: [{label, fit_features}]

  @doc """
  Creates classificator with empty data_set

  ## Examples

      iex> classificator = LearnKit.NaiveBayes.Gaussian.new
      %LearnKit.NaiveBayes.Gaussian{data_set: [], fit_data: []}

  """
  @spec new() :: %LearnKit.NaiveBayes.Gaussian{data_set: []}

  def new do
    []
    |> Gaussian.new
  end

  @doc """
  Creates classificator with data_set

  ## Parameters

    - data_set: Keyword list with labels and features in tuples

  ## Examples

      iex> classificator = LearnKit.NaiveBayes.Gaussian.new([{:a1, [[1, 2], [2, 3]]}, {:b1, [[-1, -2]]}])
      %LearnKit.NaiveBayes.Gaussian{data_set: [a1: [[1, 2], [2, 3]], b1: [[-1, -2]]], fit_data: []}

  """
  @spec new(data_set) :: %LearnKit.NaiveBayes.Gaussian{data_set: data_set}

  def new(data_set) do
    %Gaussian{data_set: data_set}
  end

  @doc """
  Add train data to classificator

  ## Parameters

    - classificator: %LearnKit.NaiveBayes.Gaussian{}
    - train data: tuple with label and feature

  ## Examples

      iex> classificator |> LearnKit.NaiveBayes.Gaussian.add_train_data({:a1, [-1, -1]})
      %LearnKit.NaiveBayes.Gaussian{data_set: [a1: [[-1, -1]]], fit_data: []}

  """
  @spec add_train_data(%LearnKit.NaiveBayes.Gaussian{data_set: data_set}, point) :: %LearnKit.NaiveBayes.Gaussian{data_set: data_set}

  def add_train_data(%Gaussian{data_set: data_set}, {key, value}) do
    features = if Keyword.has_key?(data_set, key), do: Keyword.get(data_set, key), else: []
    data_set = Keyword.put(data_set, key, [value | features])
    %Gaussian{data_set: data_set}
  end

  @doc """
  Fit train data

  ## Parameters

    - classificator: %LearnKit.NaiveBayes.Gaussian{}

  ## Examples

      iex> classificator |> LearnKit.NaiveBayes.Gaussian.fit
      %LearnKit.NaiveBayes.Gaussian{
        data_set: [a1: [[-1, -1]]],
        fit_data: [
          a1: [
            %{mean: -1.0, standard_deviation: 0.0, variance: 0.0},
            %{mean: -1.0, standard_deviation: 0.0, variance: 0.0}
          ]
        ]
      }

  """
  @spec fit(%LearnKit.NaiveBayes.Gaussian{data_set: data_set}) :: %LearnKit.NaiveBayes.Gaussian{data_set: data_set, fit_data: fit_data}

  def fit(%Gaussian{data_set: data_set}) do
    %Gaussian{data_set: data_set, fit_data: fit_data(data_set)}
  end

  @doc """
  Return probability estimates for the feature

  ## Parameters

    - classificator: %LearnKit.NaiveBayes.Gaussian{}

  ## Examples

      iex> classificator |> LearnKit.NaiveBayes.Gaussian.predict_proba([1, 2])
      {:ok, [a1: 0.0359, a2: 0.0039]}

  """
  @spec predict_proba(%LearnKit.NaiveBayes.Gaussian{fit_data: fit_data}, feature) :: {:ok, predictions}

  def predict_proba(%Gaussian{fit_data: fit_data}, feature) do
    result = fit_data |> classify_data(feature)
    {:ok, result}
  end

  @doc """
  Return exact prediction for the feature

  ## Parameters

    - classificator: %LearnKit.NaiveBayes.Gaussian{}

  ## Examples

      iex> classificator |> LearnKit.NaiveBayes.Gaussian.predict([1, 2])
      {:ok, {:a1, 0.334545454}}

  """
  @spec predict(%LearnKit.NaiveBayes.Gaussian{fit_data: fit_data}, feature) :: {:ok, prediction}

  def predict(%Gaussian{fit_data: fit_data}, feature) do
    result = fit_data |> classify_data(feature) |> Enum.sort_by(&(elem(&1, 1))) |> Enum.at(-1)
    {:ok, result}
  end

  @doc """
  Returns the mean accuracy on the given test data and labels

  ## Parameters

    - classificator: %LearnKit.NaiveBayes.Gaussian{}

  ## Examples

      iex> classificator |> LearnKit.NaiveBayes.Gaussian.score
      {:ok, 0.857143}

  """
  @spec score(%LearnKit.NaiveBayes.Gaussian{data_set: data_set, fit_data: fit_data}) :: {:ok, number}

  def score(%Gaussian{data_set: data_set, fit_data: fit_data}) do
    result = fit_data |> calc_score(data_set)
    {:ok, result}
  end
end
