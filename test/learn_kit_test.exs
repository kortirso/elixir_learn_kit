defmodule LearnKitTest do
  use ExUnit.Case
  doctest LearnKit

  test "greets the world" do
    assert LearnKit.hello() == :world
  end
end
