defmodule LearnKit.Regression.Linear.Fit do
  @moduledoc """
  Module for fit functions
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp fit_data(factors, results) do
        beta = Math.correlation(factors, results) * Math.standard_deviation(results) / Math.standard_deviation(factors)
        alpha = Math.mean(results) - beta * Math.mean(factors)
        [alpha, beta]
      end
    end
  end
end
