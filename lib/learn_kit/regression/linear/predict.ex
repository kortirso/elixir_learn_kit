defmodule LearnKit.Regression.Linear.Predict do
  @moduledoc """
  Module for fit functions
  """
  defmacro __using__(_opts) do
    quote do
      defp predict_sample(sample, [alpha, beta]) do
        sample * beta + alpha
      end
    end
  end
end
