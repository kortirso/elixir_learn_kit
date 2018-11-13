defmodule LearnKit.Knn.Predict do
  @moduledoc """
  Module for knn prediction functions
  """

  defmacro __using__(_opts) do
    quote do
      defp prediction(data_set, options) do
        calc_distances_for_features(data_set, options)
        |> sort_distances
        |> select_closest_features(options)
        |> calc_feature_weights(options)
        |> define_weight_of_labels
        |> sort_result
      end

      # select algorithm for prediction
      defp calc_distances_for_features(data_set, options) do
        case Keyword.get(options, :algorithm) do
          "brute" -> brute_algorithm(data_set, options)
          _ -> []
        end
      end

      defp sort_distances(features) do
        features
        |> Enum.sort(&(elem(&1, 0) <= elem(&2, 0)))
      end

      defp select_closest_features(features, options) do
        features
        |> Enum.take(Keyword.get(options, :k))
      end

      defp calc_feature_weights(features, options) do
        features
        |> Enum.map(fn feature ->
          Tuple.append(feature, calc_feature_weight(Keyword.get(options, :weight), elem(feature, 0)))
        end)
      end

      defp define_weight_of_labels(features) do
        features
        |> accumulate_weight_of_labels([])
      end

      defp sort_result(features) do
        features
        |> Enum.sort(&(elem(&1, 1) >= elem(&2, 1)))
        |> List.first
      end

      # brute algorithm for prediction
      defp brute_algorithm(data_set, options) do
        Keyword.keys(data_set)
        |> Enum.map(fn key ->
          Keyword.get(data_set, key)
          |> Enum.reduce([], fn feature, acc ->
            distance = feature |> calc_distance_between_features(Keyword.get(options, :feature))
            if distance == 0 do
              raise "Feature exists in train data set with label #{key}"
            end
            acc = [{distance, key} | acc]
          end)
        end)
        |> List.flatten
      end

      defp calc_distance_between_features(feature_from_data_set, feature) do
        size = length(feature_from_data_set)
        calc_distance_between_points(0, feature_from_data_set, feature, 0, size - 1)
      end

      defp calc_distance_between_points(acc, feature_from_data_set, feature, current_index, size) when current_index <= size do
        Enum.at(feature_from_data_set, current_index) - Enum.at(feature, current_index)
        |> :math.pow(2)
        |> summ(acc)
        |> calc_distance_between_points(feature_from_data_set, feature, current_index + 1, size)
      end

      defp calc_distance_between_points(acc, _feature_from_data_set, _feature, _current_index, _size) do
        acc
        |> :math.sqrt
      end

      defp summ(a, b) do
        a + b
      end

      defp calc_feature_weight(weight, distance) do
        case weight do
          "uniform" -> 1
          "distance" -> 1 / :math.pow(distance, 2)
          _ -> 1
        end
      end

      defp accumulate_weight_of_labels(features, acc) when features == [] do
        acc
      end

      defp accumulate_weight_of_labels([{_distance, key, weight} | tail], acc) do
        previous = if Keyword.has_key?(acc, key), do: Keyword.get(acc, key), else: 0
        acc = Keyword.put(acc, key, previous + weight)
        accumulate_weight_of_labels(tail, acc)
      end
    end
  end
end
