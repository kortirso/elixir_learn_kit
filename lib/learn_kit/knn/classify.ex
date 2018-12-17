defmodule LearnKit.Knn.Classify do
  @moduledoc """
  Module for knn classify functions
  """

  alias LearnKit.Math

  defmacro __using__(_opts) do
    quote do
      defp prediction(data_set, options) do
        data_set
        |> calc_distances_for_features(options)
        |> sort_distances()
        |> select_closest_features(options)
        |> check_zero_distance(options)
      end

      # select algorithm for prediction
      defp calc_distances_for_features(data_set, options) do
        case Keyword.get(options, :algorithm) do
          "brute" -> brute_algorithm(data_set, options)
          _ -> []
        end
      end

      defp sort_distances(features) do
        Enum.sort(features, &(elem(&1, 0) <= elem(&2, 0)))
      end

      defp select_closest_features(features, options) do
        Enum.take(features, Keyword.get(options, :k))
      end

      defp check_zero_distance(closest_features, options) do
        {distance, label} = Enum.at(closest_features, 0)
        cond do
          distance == 0 -> {label, 0}
          true -> select_best_label(closest_features, options)
        end
      end

      defp select_best_label(features, options) do
        features
        |> calc_feature_weights(options)
        |> accumulate_weight_of_labels([])
        |> sort_result()
      end

      defp calc_feature_weights(features, options) do
        Enum.map(features, fn feature ->
          Tuple.append(feature, calc_feature_weight(Keyword.get(options, :weight), elem(feature, 0)))
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
        |> handle_features_in_label(data_set, Keyword.get(options, :feature))
        |> List.flatten()
      end

      defp handle_features_in_label(keys, data_set, current_feature) do
        Enum.map(keys, fn key ->
          data_set
          |> Keyword.get(key)
          |> filter_features_by_size(current_feature)
          |> calc_distances_in_label(current_feature, key)
        end)
      end

      defp filter_features_by_size(features, current_feature) do
        Enum.filter(features, fn feature ->
          length(feature) == length(current_feature)
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

      #defp calc_distance_between_points(acc, feature_from_data_set, feature, current_index, size) when current_index <= size do
      #  Enum.at(feature_from_data_set, current_index) - Enum.at(feature, current_index)
      #  |> :math.pow(2)
      #  |> Math.summ(acc)
      #  |> calc_distance_between_points(feature_from_data_set, feature, current_index + 1, size)
      #end

      #defp calc_distance_between_points(acc, _, _, _, _) do
      #  :math.sqrt(acc)
      #end

      defp calc_feature_weight(weight, distance) do
        case weight do
          "uniform" -> 1
          "distance" -> 1 / :math.pow(distance, 2)
          _ -> 1
        end
      end

      defp accumulate_weight_of_labels([], acc) do
        acc
      end

      defp accumulate_weight_of_labels([{_, key, weight} | tail], acc) do
        previous = if Keyword.has_key?(acc, key), do: Keyword.get(acc, key), else: 0
        acc = Keyword.put(acc, key, previous + weight)
        accumulate_weight_of_labels(tail, acc)
      end
    end
  end
end
