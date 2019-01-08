defmodule LearnKit.Regression.Polynomial.Calculations do
  @moduledoc """
  Module for fit functions
  """

  defmacro __using__(_opts) do
    quote do
      defp do_predict(polynomial, samples) do
        Enum.map(samples, fn sample ->
          {:ok, prediction} = predict(polynomial, sample)
          prediction
        end)
      end

      defp matrix_line(1, factors, degree) do
        power_ofs = Enum.to_list(1..degree)
        [Enum.count(factors) | sum_of_x_i_with_k(power_ofs, factors)]
      end

      defp matrix_line(line, factors, degree) do
        line_factor = line - 1
        power_ofs = Enum.to_list(line_factor..(degree + line_factor))
        sum_of_x_i_with_k(power_ofs, factors)
      end

      defp matrix(factors, degree) do
        lines = Enum.to_list(1..(degree + 1))
        Enum.map(lines, fn line ->
          matrix_line(line, factors, degree)
        end)
      end

      def sum_of_x_i_with_k(ks, factors) do
        Enum.map(ks, fn factor ->
          sum_x_with_k(factors, factor, 0.0)
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
  end
end
