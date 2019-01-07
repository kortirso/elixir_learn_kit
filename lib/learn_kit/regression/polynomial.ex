defmodule LearnKit.Regression.Polynomial do
  @moduledoc """
  Module for Polynomial Regression algorithm
  """

  defstruct factors: [], results: [], coefficients: [], degree: 2

  alias LearnKit.Regression.Polynomial

  @type factors :: [number]
  @type results :: [number]
  @type coefficients :: [number]
  @type degree :: number

  @doc """
  Creates polynomial predictor with data_set

  ## Parameters

    - factors: Array of predictor variables
    - results: Array of criterion variables

  ## Examples

      iex> predictor = LearnKit.Regression.Polynomial.new([1, 2, 3, 4], [3, 6, 10, 15])
      %LearnKit.Regression.Polynomial{factors: [1, 2, 3, 4], results: [3, 6, 10, 15], coefficients: [], degree: 2}

  """
  def new(factors, results) when is_list(factors) and is_list(results) do
    %Polynomial{factors: factors, results: results}
  end

  def new(_, _), do: Polynomial.new([], [])
  def new, do: Polynomial.new([], [])

  @doc """
  Fit train data

  ## Parameters

    - predictor: %LearnKit.Regression.Polynomial{}
    - options: keyword list with options

  ## Options

    - degree: nth degree of polynomial model, default set to 2

  ## Examples

      iex> predictor = predictor |> LearnKit.Regression.Polynomial.fit
      %LearnKit.Regression.Polynomial{
        coefficients: [0.9999999999998295, 1.5000000000000853, 0.4999999999999787],
        degree: 2,
        factors: [1, 2, 3, 4],
        results: [3, 6, 10, 15]
      }

      iex> predictor = predictor |> LearnKit.Regression.Polynomial.fit([degree: 3])
      %LearnKit.Regression.Polynomial{
        coefficients: [1.0000000000081855, 1.5000000000013642, 0.5,
         8.526512829121202e-14],
        degree: 3,
        factors: [1, 2, 3, 4],
        results: [3, 6, 10, 15]
      }

  """
  def fit(%Polynomial{factors: factors, results: results}, options \\ []) do
    degree = options[:degree] || 2
    matrix = matrix(factors, degree)
    xys = x_y_matrix(factors, results, degree + 1, [])
    coefficients = matrix |> Matrix.inv() |> Matrix.mult(xys) |> List.flatten()
    %Polynomial{factors: factors, results: results, coefficients: coefficients, degree: degree}
  end

  @doc """
  Predict using the polynomial model

  ## Parameters

    - predictor: %LearnKit.Regression.Polynomial{}
    - samples: Array of variables

  ## Examples

      iex> predictor |> LearnKit.Regression.Polynomial.predict([5,6])
      {:ok, [20.999999999999723, 27.999999999999574]}

  """
  def predict(polynomial = %Polynomial{coefficients: _, degree: _}, samples)
      when is_list(samples) do
    {:ok,
     Enum.map(samples, fn sample ->
       {:ok, prediction} = predict(polynomial, sample)
       prediction
     end)}
  end

  @doc """
  Predict using the polynomial model

  ## Parameters

    - predictor: %LearnKit.Regression.Polynomial{}
    - sample: Sample variable

  ## Examples

      iex> predictor |> LearnKit.Regression.Polynomial.predict(5)
      {:ok, 20.999999999999723}

  """
  def predict(%Polynomial{coefficients: coefficients, degree: degree}, sample) do
    ordered_coefficients = coefficients |> Enum.reverse()
    {:ok, substitute_coefficients(ordered_coefficients, sample, degree, 0.0)}
  end

  defp matrix_line(1, factors, degree) do
    power_ofs = Enum.to_list(1..degree)

    [Enum.count(factors)] ++
      Enum.map(power_ofs, fn factor ->
        sum_x_with_k(factors, factor, 0.0)
      end)
  end

  defp matrix_line(line, factors, degree) do
    line_factor = line - 1
    power_ofs = Enum.to_list(line_factor..(degree + line_factor))

    Enum.map(power_ofs, fn factor ->
      sum_x_with_k(factors, factor, 0.0)
    end)
  end

  defp matrix(factors, degree) do
    lines = Enum.to_list(1..(degree + 1))

    Enum.map(lines, fn line ->
      matrix_line(line, factors, degree)
    end)
  end

  defp substitute_coefficients([], _, _, sum), do: sum

  defp substitute_coefficients([coefficient | tail], x, k, sum) do
    sum = sum + :math.pow(x, k) * coefficient
    substitute_coefficients(tail, x, k - 1, sum)
  end

  defp sum_x_with_k([x | tail], k, sum) do
    sum = sum + :math.pow(x, k)
    sum_x_with_k(tail, k, sum)
  end

  defp sum_x_with_k([], _, sum), do: sum

  defp sum_x_y_with_k([], [], _degree, sum), do: [sum]

  defp sum_x_y_with_k([x | xtail], [y | ytail], degree, sum) do
    exponent = degree - 1
    sum = sum + :math.pow(x, exponent) * y
    sum_x_y_with_k(xtail, ytail, degree, sum)
  end

  def x_y_matrix(_, _, 0, matrix), do: matrix |> Enum.reverse()

  def x_y_matrix(xs, ys, degree, matrix) do
    matrix = matrix ++ [sum_x_y_with_k(xs, ys, degree, 0.0)]
    x_y_matrix(xs, ys, degree - 1, matrix)
  end
end
