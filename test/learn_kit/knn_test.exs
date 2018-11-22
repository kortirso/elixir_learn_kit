defmodule LearnKit.KnnTest do
  use ExUnit.Case

  alias LearnKit.Knn

  test "create new knn classificator with empty data set" do
    assert %Knn{data_set: data_set} = Knn.new

    assert data_set == []
  end

  test "add train data to classificator" do
    %Knn{data_set: data_set} = Knn.new
                    |> Knn.add_train_data({:a1, [1, 2]})
                    |> Knn.add_train_data({:a1, [1, 3]})
                    |> Knn.add_train_data({:b1, [2, 3]})

    assert data_set == [b1: [[2, 3]], a1: [[1, 3], [1, 2]]]
  end

  test "classify new feature" do
    classificator = Knn.new
                    |> Knn.add_train_data({:a1, [-1, -1]})
                    |> Knn.add_train_data({:a1, [-2, -1]})
                    |> Knn.add_train_data({:a1, [-3, -2]})
                    |> Knn.add_train_data({:a2, [1, 1]})
                    |> Knn.add_train_data({:a2, [2, 1]})
                    |> Knn.add_train_data({:a2, [3, 2]})
                    |> Knn.add_train_data({:a2, [-2, -2]})

    assert {:ok, :a1} = Knn.classify(classificator, [feature: [-1, -2], k: 3, weight: "distance"])
  end
end
