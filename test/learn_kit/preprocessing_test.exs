defmodule LearnKit.PreprocessingTest do
  use ExUnit.Case

  alias LearnKit.Preprocessing

  describe "for invalid data" do
    test "use preprocessor with invalid data" do
      assert_raise FunctionClauseError, fn ->
        Preprocessing.normalize("")
      end
    end

    test "use preprocessor with invalid options" do
      assert_raise FunctionClauseError, fn ->
        Preprocessing.normalize([[1, 2], [3, 4], [5, 6]], "")
      end
    end
  end

  describe "for valid data" do
    test "normalize data set with minimax normalization" do
      result = Preprocessing.normalize([[1, 2], [3, 4], [5, 6]])

      assert result == [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    end

    test "normalize data set with z normalization" do
      result = Preprocessing.normalize([[1, 2], [3, 4], [5, 6]], [type: "z_normalization"])

      assert result == [[-1.224744871391589, -1.224744871391589], [0.0, 0.0], [1.224744871391589, 1.224744871391589]]
    end
  end
end
