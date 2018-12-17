defmodule LearnKit.KnnTest do
  use ExUnit.Case

  alias LearnKit.Knn

  setup_all do
    {:ok, classifier: Knn.new([{:a1, [[-1, -1], [-2, -1], [-3, -2]]}, {:b1, [[1, 1], [2, 1], [3, 2], [-2, -2]]}])}
  end

  describe "for invalid data" do
    test "create new classifier with invalid data" do
      assert_raise FunctionClauseError, fn ->
        Knn.new("")
      end
    end

    test "add train data in invalid format", state do
      assert_raise FunctionClauseError, fn ->
        Knn.add_train_data(state[:classifier], {:something_valid, "invalid"})
      end
    end

    test "classify without options", state do
      assert_raise FunctionClauseError, fn ->
        Knn.classify(state[:classifier], "")
      end
    end

    test "classify with empty options", state do
      assert {:error, "Feature option is required"} = Knn.classify(state[:classifier], [])
    end

    test "classify with invalid feature", state do
      assert {:error, "Feature option must be presented as array"} = Knn.classify(state[:classifier], [feature: "1"])
    end

    test "classify with invalid k", state do
      assert {:error, "K option must be positive integer"} = Knn.classify(state[:classifier], [feature: [-1, -2], k: -2])
    end
  end

  describe "for valid data" do
    test "create new knn classifier with empty data set" do
      assert %Knn{data_set: data_set} = Knn.new

      assert data_set == []
    end

    test "add train data to classifier" do
      %Knn{data_set: data_set} =
        Knn.new
        |> Knn.add_train_data({:a1, [1, 2]})
        |> Knn.add_train_data({:a1, [1, 3]})
        |> Knn.add_train_data({:b1, [2, 3]})

      assert data_set == [b1: [[2, 3]], a1: [[1, 3], [1, 2]]]
    end

    test "classify new feature", state do
      assert {:ok, :a1} = Knn.classify(state[:classifier], [feature: [-1, -2], k: 3, weight: "distance"])
    end

    test "classify new feature, for existed point", state do
      assert {:ok, :b1} = Knn.classify(state[:classifier], [feature: [-2, -2], k: 3, weight: "uniform"])
    end
  end
end
