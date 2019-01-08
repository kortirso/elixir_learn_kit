defmodule LearnKit.MathTest do
  use ExUnit.Case
  alias LearnKit.Math

  test "calculate sum" do
    assert 3 = Math.summ(1, 2)
  end

  test "calculate mean" do
    assert 2.0 = Math.mean([1, 2, 3])
  end

  test "calculate variance" do
    assert 1.25 = Math.variance([1, 2, 3, 4])
  end

  test "calculate variance, with calculated mean" do
    assert 1.25 = Math.variance([1, 2, 3, 4], 2.5)
  end

  test "calculate standard deviation" do
    assert 0.5 = Math.standard_deviation([1, 2])
  end

  test "calculate standard deviation from variance" do
    assert 1.118033988749895 = Math.standard_deviation_from_variance(1.25)
  end

  test "calculate division" do
    assert 5.0 = Math.division(10, 2)
  end

  test "calculate covariance" do
    assert 5.5 = Math.covariance([1, 2, 3], [14, 17, 25])
  end

  test "calculate correlation" do
    assert 0.9672471299049061 = Math.correlation([1, 2, 3], [14, 17, 25])
  end

  test "transposing a matrix" do
    assert [[1, 3, 5], [2, 4, 6]] = Math.transpose([[1, 2], [3, 4], [5, 6]])
  end

  test "scalar multiplication with matrix" do
    assert [50, 60] = Math.scalar_multiply(10, [5, 6])
  end

  test "vector subtraction" do
    assert [5, 45, 20] = Math.vector_subtraction([40, 50, 60], [35, 5, 40])
  end
end
