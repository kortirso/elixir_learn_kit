defmodule LearnKit.Knn do
  @moduledoc """
  Module for k-nearest neighbours (knn) algorithm
  """

  defstruct data_set: []

  alias LearnKit.{Knn}

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
end
