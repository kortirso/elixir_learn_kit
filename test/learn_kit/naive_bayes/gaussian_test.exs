defmodule LearnKit.NaiveBayes.GaussianTest do
  use ExUnit.Case

  alias LearnKit.{NaiveBayes}

  test "create new knn classificator with empty data set" do
    assert %NaiveBayes.Gaussian{data_set: data_set} = NaiveBayes.Gaussian.new

    assert data_set == []
  end

  test "add train data to classificator" do
    %NaiveBayes.Gaussian{data_set: data_set} = NaiveBayes.Gaussian.new
                    |> NaiveBayes.Gaussian.add_train_data({:a1, [1, 2]})
                    |> NaiveBayes.Gaussian.add_train_data({:a1, [1, 3]})
                    |> NaiveBayes.Gaussian.add_train_data({:b1, [2, 3]})

    assert data_set == [b1: [[2, 3]], a1: [[1, 3], [1, 2]]]
  end
end
