defmodule LearnKit.MathTest do
  use ExUnit.Case

  alias LearnKit.{Math}

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
end
