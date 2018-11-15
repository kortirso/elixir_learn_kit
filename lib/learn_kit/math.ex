defmodule LearnKit.Math do
  @moduledoc """
  Math module
  """

  @doc """
  Calculate the mean from a list of numbers

  ## Examples

      iex> LearnKit.Math.mean([])
      nil

      iex> LearnKit.Math.mean([1, 2, 3])
      2.0

  """
  @spec mean(list) :: number

  def mean(list) when is_list(list), do: mean(list, 0, 0)

  defp mean([], 0, 0), do: nil

  defp mean([], sum, number), do: sum / number

  defp mean([head | tail], sum, number) do
    mean(tail, sum + head, number + 1)
  end

  @doc """
  Calculate variance from a list of numbers

  ## Examples

      iex> LearnKit.Math.variance([])
      nil

      iex> LearnKit.Math.variance([1, 2, 3, 4])
      1.25

  """
  @spec variance(list) :: number

  def variance([]), do: nil

  def variance(list) when is_list(list) do
    list_mean = mean(list)
    list
    |> variance(list_mean)
  end

  @doc """
  Calculate variance from a list of numbers, with calculated mean

  ## Examples

      iex> LearnKit.Math.variance([1, 2, 3, 4], 2.5)
      1.25

  """
  @spec variance(list, number) :: number

  def variance(list, list_mean) when is_list(list) do
    list
    |> Enum.map(fn x ->
      list_mean - x
      |> :math.pow(2)
    end)
    |> mean
  end

  @doc """
  Calculate standard deviation from a list of numbers

  ## Examples

      iex> LearnKit.Math.standard_deviation([])
      nil

      iex> LearnKit.Math.standard_deviation([1, 2])
      0.5

  """
  @spec standard_deviation(list) :: number

  def standard_deviation([]), do: nil

  def standard_deviation(list) when is_list(list) do
    list
    |> variance
    |> :math.sqrt
  end

  @doc """
  Calculate standard deviation from a list of numbers, with calculated variance

  ## Examples

      iex> LearnKit.Math.standard_deviation_from_variance(1.25)
      1.118033988749895

  """
  @spec standard_deviation_from_variance(number) :: number

  def standard_deviation_from_variance(list_variance) do
    list_variance
    |> :math.sqrt
  end
end
