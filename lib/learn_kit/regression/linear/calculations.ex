defmodule LearnKit.Regression.Linear.Calculations do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Math
  alias LearnKit.Regression.Linear

  defmacro __using__(_opts) do
    quote do
      defp do_fit(method, %Linear{factors: factors, results: results})
           when method == "gradient descent" do
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
        beta =
          Math.correlation(factors, results) * Math.standard_deviation(results) /
            Math.standard_deviation(factors)

        alpha = Math.mean(results) - beta * Math.mean(factors)
        [alpha, beta]
      end

      defp do_predict(linear, samples) do
        Enum.map(samples, fn sample ->
          {:ok, prediction} = predict(linear, sample)
          prediction
        end)
      end

      defp squared_error_gradient(linear, x, y) do
        error_variable = prediction_error(linear, x, y)

        [
          -2 * error_variable,
          -2 * error_variable * x
        ]
      end

      defp gradient_descent_iteration(_, _, min_theta, _, _, iterations_with_no_improvement)
           when iterations_with_no_improvement >= 100,
           do: min_theta

      defp gradient_descent_iteration(
             theta,
             alpha,
             min_theta,
             min_value,
             data,
             iterations_with_no_improvement
           ) do
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
            gradient_i = squared_error_gradient(%Linear{coefficients: theta}, xi, yi)
            acc |> Math.vector_subtraction(alpha |> Math.scalar_multiply(gradient_i))
          end)

        gradient_descent_iteration(
          theta,
          alpha,
          min_theta,
          min_value,
          data,
          iterations_with_no_improvement
        )
      end

      defp check_value(data, min_value, theta, min_theta, iterations_with_no_improvement, alpha) do
        value =
          Enum.reduce(data, 0, fn {xi, yi}, acc ->
            acc + squared_prediction_error(%Linear{coefficients: theta}, xi, yi)
          end)

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
