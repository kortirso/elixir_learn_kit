defmodule LearnKit.Regression.Linear do
  @moduledoc """
  Module for Linear Regression algorithm
  """

  defstruct factors: [], results: [], coefficients: []

  alias LearnKit.Regression.Linear

  use Linear.Fit
  use Linear.Predict

  @type factors :: [number]
  @type results :: [number]
  @type coefficients :: [number]

  @doc """
  Creates predictor with empty data_set

  ## Examples

      iex> predictor = LearnKit.Regression.Linear.new
      %LearnKit.Regression.Linear{factors: [], results: [], coefficients: []}

  """
  @spec new() :: %LearnKit.Regression.Linear{factors: [], results: [], coefficients: []}

  def new do
    Linear.new([], [])
  end

  @doc """
  Creates predictor with data_set

  ## Parameters

    - factors: Array of predictor variables
    - results: Array of criterion variables

  ## Examples

      iex> predictor = LearnKit.Regression.Linear.new([1, 2, 3, 4], [3, 6, 10, 15])
      %LearnKit.Regression.Linear{factors: [1, 2, 3, 4], results: [3, 6, 10, 15], coefficients: []}

  """
  @spec new(factors, results) :: %LearnKit.Regression.Linear{factors: factors, results: results, coefficients: []}

  def new(factors, results) do
    %Linear{factors: factors, results: results}
  end

  @doc """
  Fit train data

  ## Parameters

    - predictor: %LearnKit.Regression.Linear{}

  ## Examples

      iex> predictor = predictor |> LearnKit.Regression.Linear.fit
      %LearnKit.Regression.Linear{
        coefficients: [-1.5, 4.0],
        factors: [1, 2, 3, 4],
        results: [3, 6, 10, 15]
      }

  """
  @spec fit(%LearnKit.Regression.Linear{factors: factors, results: results}) :: %LearnKit.Regression.Linear{factors: factors, results: results, coefficients: coefficients}

  def fit(%Linear{factors: factors, results: results}) do
    %Linear{factors: factors, results: results, coefficients: fit_data(factors, results)}
  end

  @doc """
  Predict using the linear model

  ## Parameters

    - predictor: %LearnKit.Regression.Linear{}
    - samples: Array of variables

  ## Examples

      iex> predictor |> LearnKit.Regression.Linear.predict([4, 8, 13])
      [14.5, 30.5, 50.5]

  """
  @spec predict(%LearnKit.Regression.Linear{coefficients: coefficients}, list) :: list

  def predict(%Linear{coefficients: coefficients}, samples) do
    IO.inspect coefficients
    samples
    |> Enum.map(fn sample -> predict_sample(sample, coefficients) end)
  end
end
