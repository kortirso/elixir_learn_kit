defmodule LearnKit.Regression.Linear do
  @moduledoc """
  Module for Linear Regression algorithm
  """

  defstruct factors: [], results: []

  alias LearnKit.Regression.Linear

  @type factors :: [number]
  @type results :: [number]

  @doc """
  Creates classificator with empty data_set

  ## Examples

      iex> classificator = LearnKit.Regression.Linear.new
      %LearnKit.Regression.Linear{factors: [], results: []}

  """
  @spec new() :: %LearnKit.Regression.Linear{factors: [], results: []}

  def new do
    Linear.new([], [])
  end

  @doc """
  Creates classificator with data_set

  ## Parameters

    - data_set: Keyword list with labels and features in tuples

  ## Examples

      iex> classificator = LearnKit.Regression.Linear.new([1, 2, 3, 4], [2, 3, 4, 5])
      %LearnKit.Regression.Linear{factors: [1, 2, 3, 4], results: [2, 3, 4, 5]}

  """
  @spec new(factors, results) :: %LearnKit.Regression.Linear{factors: factors, results: results}

  def new(factors, results) do
    %Linear{factors: factors, results: results}
  end
end
