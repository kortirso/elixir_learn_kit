defmodule LearnKit.Knn do
  @moduledoc """
  Module for k-nearest neighbours (knn) algorithm
  """

  defstruct data_set: []

  alias LearnKit.{Knn}

  use Knn.Predict

  @doc """
  Creates classificator with empty data_set
  """
  def new do
    []
    |> Knn.new
  end

  @doc """
  Creates classificator with data_set
  """
  def new(data_set) do
    %Knn{data_set: data_set}
  end

  @doc """
  Add train data to classificator
  """
  def add_train_data(%Knn{data_set: data_set}, {key, value}) do
    features = if Keyword.has_key?(data_set, key), do: Keyword.get(data_set, key), else: []
    data_set = Keyword.put(data_set, key, [value | features])
    %Knn{data_set: data_set}
  end

  @doc """
  Predict label of the feature
  Available options:
  feature - feature for prediction, required, example: [1, 2, 3]
  k - number of nearest neighbours, default is 3, optional
  algorithm - brute, optional
  weight - uniform/distance, default is uniform, optional
  """
  def predict(%Knn{data_set: data_set}, options \\ []) do
    try do
      unless Keyword.has_key?(options, :feature) do
        raise "Feature option is required"
      end
      # modification of options
      options = Keyword.merge([k: 3, algorithm: "brute", weight: "uniform"], options)
      # prediction
      [{label, weight}] = prediction(data_set, options)
      {:ok, "Feature has label #{label} with weight #{weight}"}
    rescue
      error -> {:error, error.message}
    end
  end
end
