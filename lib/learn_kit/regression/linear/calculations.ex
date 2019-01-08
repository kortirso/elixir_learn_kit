defmodule LearnKit.Regression.Linear.Calculations do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.{Math, Regression.Linear}

  defmacro __using__(_opts) do
    quote do
      defp do_fit("gradient descent", %Linear{factors: factors, results: results}) do
        gradient_descent_iteration(
          [:rand.uniform(), :rand.uniform()],
          0.0001,
          nil,
          1_000_000,
          Enum.zip(factors, results),
          0
        )
      end

      defp do_fit(_, %Linear{factors: factors, results: results}) do
        beta = calc_beta(factors, results)
        alpha = Math.mean(results) - beta * Math.mean(factors)
        [alpha, beta]
      end

      defp do_predict(linear, samples) do
        Enum.map(samples, fn sample ->
          {:ok, prediction} = predict(linear, sample)
          prediction
        end)
      end

      defp calc_beta(factors, results) do
        Math.correlation(factors, results) * Math.standard_deviation(results) / Math.standard_deviation(factors)
      end

      defp squared_error_gradient(linear, x, y) do
        error_variable = prediction_error(linear, x, y)
        [
          -2 * error_variable,
          -2 * error_variable * x
        ]
      end

      defp gradient_descent_iteration(_, _, min_theta, _, _, no_improve_step) when no_improve_step >= 100, do: min_theta

      defp gradient_descent_iteration(theta, alpha, min_theta, min_value, data, no_improve_step) do
        [
          min_theta,
          min_value,
          no_improve_step,
          alpha
        ] = check_value(data, min_value, theta, min_theta, no_improve_step, alpha)

        calc_new_theta(data, theta, alpha)
        |> gradient_descent_iteration(alpha, min_theta, min_value, data, no_improve_step)
      end

      defp calc_new_theta(data, theta, alpha) do
        data
        |> Enum.shuffle()
        |> Enum.reduce(theta, fn {xi, yi}, acc ->
          gradient_i = squared_error_gradient(%Linear{coefficients: theta}, xi, yi)
          acc |> Math.vector_subtraction(alpha |> Math.scalar_multiply(gradient_i))
        end)
      end

      defp check_value(data, min_value, theta, min_theta, no_improve_step, alpha) do
        value = calc_new_value(data, theta)
        cond do
          value < min_value -> [theta, value, 0, 0.0001]
          true -> [min_theta, min_value, no_improve_step + 1, alpha * 0.9]
        end
      end

      defp calc_new_value(data, theta) do
        Enum.reduce(data, 0, fn {xi, yi}, acc ->
          acc + squared_prediction_error(%Linear{coefficients: theta}, xi, yi)
        end)
      end
    end
  end
end
