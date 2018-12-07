defmodule LearnKit.Regression.Linear.Calculations do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp fit_data(method, factors, results) when method == "gradient descent" do
        gradient_descent_iteration([:rand.uniform, :rand.uniform], 0.0001, nil, 1000000, Enum.zip(factors, results), 0)
      end

      defp fit_data(_, factors, results) do
        beta = Math.correlation(factors, results) * Math.standard_deviation(results) / Math.standard_deviation(factors)
        alpha = Math.mean(results) - beta * Math.mean(factors)
        [alpha, beta]
      end

      defp predict_sample(sample, [alpha, beta]) do
        sample * beta + alpha
      end

      defp calculate_score([], _, _), do: raise("There was no fit for model")

      defp calculate_score(coefficients, factors, results) do
        1.0 - sum_of_squared_errors(coefficients, factors, results) / total_sum_of_squares(results)
      end

      defp total_sum_of_squares(list) do
        mean_list = Math.mean(list)
        Enum.reduce(list, 0, fn x, acc -> acc + :math.pow(x - mean_list, 2) end)
      end

      defp sum_of_squared_errors(coefficients, factors, results) do
        Enum.zip(factors, results)
        |> Enum.reduce(0, fn {xi, yi}, acc -> acc + squared_prediction_error(coefficients, xi, yi) end)
      end

      defp squared_prediction_error(coefficients, x, y) do
        coefficients
        |> prediction_error(x, y)
        |> :math.pow(2)
      end

      defp squared_error_gradient(coefficients, x, y) do
        error_variable = prediction_error(coefficients, x, y)
        [
          -2 * error_variable,
          -2 * error_variable * x
        ]
      end

      defp prediction_error(coefficients, x, y) do
        y - predict_sample(x, coefficients)
      end

      defp gradient_descent_iteration(_, _, min_theta, _, _, iterations_with_no_improvement) when iterations_with_no_improvement >= 100, do: min_theta

      defp gradient_descent_iteration(theta, alpha, min_theta, min_value, data, iterations_with_no_improvement) do
        [
          min_theta,
          min_value,
          iterations_with_no_improvement,
          alpha
        ] = check_value(data, min_value, theta, min_theta, iterations_with_no_improvement, alpha)

        theta =
          data
          |> Enum.shuffle()
          |> Enum.reduce(theta, fn {xi, yi}, acc ->
            gradient_i = squared_error_gradient(acc, xi, yi)
            acc |> Math.vector_subtraction(alpha |> Math.scalar_multiply(gradient_i))
          end)
        gradient_descent_iteration(theta, alpha, min_theta, min_value, data, iterations_with_no_improvement)
      end

      defp check_value(data, min_value, theta, min_theta, iterations_with_no_improvement, alpha) do
        value = Enum.reduce(data, 0, fn {xi, yi}, acc -> acc + squared_prediction_error(theta, xi, yi) end)
        cond do
          value < min_value ->
            [theta, value, 0, 0.0001]

          true ->
            [min_theta, min_value, iterations_with_no_improvement + 1, alpha * 0.9]
        end
      end
    end
  end
end
