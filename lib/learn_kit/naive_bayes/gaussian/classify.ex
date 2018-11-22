defmodule LearnKit.NaiveBayes.Gaussian.Classify do
  @moduledoc """
  Module for prediction functions
  """
  defmacro __using__(_opts) do
    quote do
      # classify data
      # returns data like [label1: 0.03592747361085857, label2: 0.00399309643713954]
      defp classify_data(fit_data, feature) do
        labels_count = fit_data |> Keyword.keys |> length
        fit_data
        |> Enum.map(fn {label, fit_results} ->
          {label, class_probability(labels_count, feature, fit_results)}
        end)
      end

      # compute the final naive Bayesian probability for a given set of features being a part of a given label
      defp class_probability(labels_count, feature, fit_results) do
        class_fraction = 1.0 / labels_count
        feature_bayes = feature_mult(feature, fit_results, 1.0, 0)
        feature_bayes * class_fraction
        |> Float.round(10)
      end

      # multiply together the feature probabilities for all of the features in a label for given values
      defp feature_mult([], _, acc, _), do: acc

      defp feature_mult([head | tail], fit_results, acc, index) do
        acc = acc * feature_probability(index, head, fit_results)
        feature_mult(tail, fit_results, acc, index + 1)
      end

      defp feature_probability(index, value, fit_results) do
        # select result from training
        fit_result = Enum.at(fit_results, index)
        # deal with the edge case of a 0 standard deviation
        if fit_result.standard_deviation == 0 do
          if fit_result.mean == value, do: 1.0, else: 0.0
        else
        # calculate the gaussian probability
          exp = - :math.pow(value - fit_result.mean, 2) / (2 * fit_result.variance)
          :math.exp(exp) / :math.sqrt(2 * :math.pi * fit_result.variance)
        end
      end
    end
  end
end
