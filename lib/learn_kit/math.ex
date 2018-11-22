defmodule LearnKit.Math do
  @moduledoc """
  Math module
  """

  @type row :: [number]
  @type matrix :: [row]

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
    |> Enum.map(fn x -> :math.pow(list_mean - x, 2) end)
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

  @doc """
  Transposing a matrix

  ## Examples

      iex> LearnKit.Math.transpose([[1, 2], [3, 4], [5, 6]])
      [[1, 3, 5], [2, 4, 6]]

  """
  @spec transpose(matrix) :: matrix

  def transpose(m) do
    swap_rows_cols(m)
  end

  defp swap_rows_cols([head | _]) when head == [], do: []

  defp swap_rows_cols(rows) do
    firsts = Enum.map(rows, fn(x) -> hd(x) end)
    others = Enum.map(rows, fn(x) -> tl(x) end)
    [firsts | swap_rows_cols(others)]
  end

  @doc """
  Division for 2 elements

  ## Examples

      iex> LearnKit.Math.division(10, 2)
      5

  """
  @spec division(number, number) :: number

  def division(x, y) when y != 0 do
    x / y
  end

  @doc """
  Calculate the covariance of two lists

  ## Examples

      iex> LearnKit.Math.covariance([1, 2, 3], [14, 17, 25])
      5.5

  """
  @spec covariance(list,list) :: number

  def covariance(x, y) when length(x) == length(y) do
    mean_x = mean(x)
    mean_y = mean(y)
    size = length(x)

    Enum.zip(x, y)
    |> Enum.map(fn {xi, yi} -> (xi - mean_x) * (yi - mean_y) end)
    |> Enum.sum
    |> division(size - 1)
  end

  @doc """
  Correlation of two lists

  ## Examples

      iex> LearnKit.Math.correlation([1, 2, 3], [14, 17, 25])
      0.9672471299049061

  """
  @spec correlation(list, list) :: number

  def correlation(x, y) when length(x) == length(y) do
    mean_x = mean(x)
    mean_y = mean(y)

    divider = Enum.zip(x, y)
            |> Enum.map(fn {xi, yi} -> (xi - mean_x) * (yi - mean_y) end)
            |> Enum.sum
    denom_x = x
              |> Enum.map(fn xi -> :math.pow(xi - mean_x, 2) end)
              |> Enum.sum
    denom_y = y
              |> Enum.map(fn yi -> :math.pow(yi - mean_y, 2) end)
              |> Enum.sum

    divider / :math.sqrt(denom_x * denom_y)
  end
end
