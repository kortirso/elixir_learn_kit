defmodule LearnKit.Regression.Linear do
  @moduledoc """
  Module for Linear Regression algorithm
  """

  defstruct factors: [], results: [], coefficients: []

  alias LearnKit.Regression.Linear

  use Linear.Fit

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

      iex> predictor = LearnKit.Regression.Linear.new([1, 2, 3, 4], [2, 3, 4, 5])
      %LearnKit.Regression.Linear{factors: [1, 2, 3, 4], results: [2, 3, 4, 5], coefficients: []}

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

      iex> predictor |> LearnKit.Regression.Linear.fit
      %LearnKit.Regression.Linear{
        coefficients: [1.0, 1.0],
        factors: [1, 2, 3, 4],
        results: [2, 3, 4, 5]
      }

  """
  @spec fit(%LearnKit.Regression.Linear{factors: factors, results: results}) :: %LearnKit.Regression.Linear{factors: factors, results: results, coefficients: coefficients}

  def fit(%Linear{factors: factors, results: results}) do
    %Linear{factors: factors, results: results, coefficients: fit_data(factors, results)}
  end
end
