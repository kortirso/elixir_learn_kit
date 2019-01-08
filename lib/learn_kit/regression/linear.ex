defmodule LearnKit.Regression.Linear do
  @moduledoc """
  Module for Linear Regression algorithm
  """

  defstruct factors: [], results: [], coefficients: []

  alias LearnKit.Regression.Linear

  use Linear.Calculations
  use LearnKit.Regression.Score

  @type factors :: [number]
  @type results :: [number]
  @type coefficients :: [number]

  @doc """
  Creates predictor with empty data_set

  ## Examples

      iex> predictor = LearnKit.Regression.Linear.new
      %LearnKit.Regression.Linear{factors: [], results: [], coefficients: []}

  """
  @spec new() :: %Linear{factors: [], results: [], coefficients: []}

  def new, do: Linear.new([], [])

  @doc """
  Creates predictor with data_set

  ## Parameters

    - factors: Array of predictor variables
    - results: Array of criterion variables

  ## Examples

      iex> predictor = LearnKit.Regression.Linear.new([1, 2, 3, 4], [3, 6, 10, 15])
      %LearnKit.Regression.Linear{factors: [1, 2, 3, 4], results: [3, 6, 10, 15], coefficients: []}

  """
  @spec new(factors, results) :: %Linear{factors: factors, results: results, coefficients: []}

  def new(factors, results) when is_list(factors) and is_list(results),
    do: %Linear{factors: factors, results: results}

  @doc """
  Fit train data

  ## Parameters

    - predictor: %LearnKit.Regression.Linear{}
    - options: keyword list with options

  ## Options

    - method: method for fit, "least squares"/"gradient descent", default is "least squares", optional

  ## Examples

      iex> predictor = predictor |> LearnKit.Regression.Linear.fit
      %LearnKit.Regression.Linear{
        coefficients: [-1.5, 4.0],
        factors: [1, 2, 3, 4],
        results: [3, 6, 10, 15]
      }

      iex> predictor = predictor |> LearnKit.Regression.Linear.fit([method: "gradient descent"])
      %LearnKit.Regression.Linear{
        coefficients: [-1.4975720508482548, 3.9992148848913356],
        factors: [1, 2, 3, 4],
        results: [3, 6, 10, 15]
      }

  """
  @spec fit(%Linear{factors: factors, results: results}) :: %Linear{
          factors: factors,
          results: results,
          coefficients: coefficients
        }

  def fit(linear = %Linear{factors: factors, results: results}, options \\ [])
      when is_list(options) do
    coefficients =
      Keyword.merge([method: ""], options)
      |> define_method_for_fit()
      |> do_fit(linear)

    %Linear{factors: factors, results: results, coefficients: coefficients}
  end

  defp define_method_for_fit(options) do
    case options[:method] do
      "gradient descent" -> "gradient descent"
      _ -> ""
    end
  end

  @doc """
  Predict using the linear model

  ## Parameters

    - predictor: %LearnKit.Regression.Linear{}
    - samples: Array of variables

  ## Examples

      iex> predictor |> LearnKit.Regression.Linear.predict([4, 8, 13])
      {:ok, [14.5, 30.5, 50.5]}

  """
  @spec predict(%Linear{coefficients: coefficients}, list) :: {:ok, list}

  def predict(linear = %Linear{coefficients: _}, samples) when is_list(samples) do
    {
      :ok,
      Enum.map(samples, fn sample -> predict(linear, sample) end)
    }
  end

  @doc """
  Predict using the linear model

  ## Parameters

    - predictor: %LearnKit.Regression.Linear{}
    - sample: Sample variable

  ## Examples

      iex> predictor |> LearnKit.Regression.Linear.predict(4)
      {:ok, 14.5}

  """
  @spec predict(%Linear{coefficients: coefficients}, list) :: {:ok, list}

  def predict(%Linear{coefficients: [alpha, beta]}, sample) do
    sample * beta + alpha
  end

  @doc """
  Returns the coefficient of determination R^2 of the prediction

  ## Parameters

    - predictor: %LearnKit.Regression.Linear{}

  ## Examples

      iex> predictor |> LearnKit.Regression.Linear.score
      {:ok, 0.9876543209876543}

  """
  @spec score(%Linear{factors: factors, results: results, coefficients: coefficients}) ::
          {:ok, number}

  def score(linear = %Linear{factors: _, results: _, coefficients: _}) do
    {
      :ok,
      calculate_score(linear)
    }
  end
end
