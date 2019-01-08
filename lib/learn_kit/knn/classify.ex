defmodule LearnKit.Knn.Classify do
  @moduledoc """
  Module for knn classify functions
  """

  alias LearnKit.{Preprocessing, Math}

  defmacro __using__(_opts) do
    quote do
      defp prediction(data_set, options) do
        data_set
        |> filter_features_by_size(options[:feature])
        |> check_normalization(options)
        |> calc_distances_for_features(options)
        |> sort_distances()
        |> select_closest_features(options)
        |> check_zero_distance(options)
      end

      # knn uses only features with the same size as current feature
      defp filter_features_by_size(data_set, current_feature) do
        Enum.map(data_set, fn {key, features} ->
          {
            key,
            Enum.filter(features, fn feature -> length(feature) == length(current_feature) end)
          }
        end)
      end

      # normalize features
      defp check_normalization(data_set, options) do
        type = options[:normalization]
        case type do
          t when t in ["minimax", "z_normalization"] -> normalize(data_set, type)
          _ -> data_set
        end
      end

      # select algorithm for prediction
      defp calc_distances_for_features(data_set, options) do
        case options[:algorithm] do
          "brute" -> brute_algorithm(data_set, options)
          _ -> []
        end
      end

      # sort distances
      defp sort_distances(features), do: Enum.sort(features, &(elem(&1, 0) <= elem(&2, 0)))

      # take closest features
      defp select_closest_features(features, options), do: Enum.take(features, options[:k])

      # check existeness of current feature in data set
      defp check_zero_distance(closest_features, options) do
        {distance, label} = Enum.at(closest_features, 0)
        cond do
          distance == 0 -> {label, 0}
          true -> select_best_label(closest_features, options)
        end
      end

      # select best result based on weights
      defp select_best_label(features, options) do
        features
        |> calc_feature_weights(options)
        |> accumulate_weight_of_labels([])
        |> sort_result()
      end

      # normalize each feature
      defp normalize(data_set, type) do
        coefficients = find_coefficients_for_normalization(data_set, type)
        Enum.map(data_set, fn {key, features} ->
          {
            key,
            Enum.map(features, fn feature -> Preprocessing.normalize_feature(feature, coefficients, type) end)
          }
        end)
      end

      # find coefficients for normalization
      defp find_coefficients_for_normalization(data_set, type) do
        Enum.reduce(data_set, [], fn {_, features}, acc ->
          Enum.reduce(features, acc, fn feature, acc -> [feature | acc] end)
        end)
        |> Preprocessing.coefficients(type)
      end

      defp calc_feature_weights(features, options) do
        Enum.map(features, fn feature ->
          Tuple.append(feature, calc_feature_weight(options[:weight], elem(feature, 0)))
        end)
      end

      defp sort_result(features) do
        features
        |> Enum.sort(&(elem(&1, 1) >= elem(&2, 1)))
        |> List.first()
      end

      # brute algorithm for prediction
      defp brute_algorithm(data_set, options) do
        data_set
        |> Keyword.keys()
        |> handle_features_in_label(data_set, options[:feature])
        |> List.flatten()
      end

      defp handle_features_in_label(keys, data_set, current_feature) do
        Enum.map(keys, fn key ->
          data_set[key]
          |> calc_distances_in_label(current_feature, key)
        end)
      end

      defp calc_distances_in_label(features, current_feature, key) do
        Enum.reduce(features, [], fn feature, acc ->
          distance = calc_distance_between_features(feature, current_feature)
          acc = [{distance, key} | acc]
        end)
      end

      defp calc_distance_between_features(feature_from_data_set, feature) do
        Enum.zip(feature_from_data_set, feature)
        |> calc_distance_between_points()
        |> :math.sqrt()
      end

      defp calc_distance_between_points(list) do
        Enum.reduce(list, 0, fn {xi, yi}, acc ->
          xi - yi
          |> :math.pow(2)
          |> Math.summ(acc)
        end)
      end

      defp calc_feature_weight(weight, distance) do
        case weight do
          "uniform" -> 1
          "distance" -> 1 / :math.pow(distance, 2)
          _ -> 1
        end
      end

      defp accumulate_weight_of_labels([], acc), do: acc

      defp accumulate_weight_of_labels([{_, key, weight} | tail], acc) do
        previous = if Keyword.has_key?(acc, key), do: acc[key], else: 0
        acc = Keyword.put(acc, key, previous + weight)
        accumulate_weight_of_labels(tail, acc)
      end
    end
  end
end
