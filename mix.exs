defmodule LearnKit.MixProject do
  use Mix.Project

  @description """
    Elixir package for machine learning
  """

  def project do
    [
      app: :learn_kit,
      version: "0.1.6",
      elixir: "~> 1.7",
      name: "LearnKit",
      description: @description,
      source_url: "https://github.com/kortirso/elixir_learn_kit",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:ex_doc, "~> 0.19", only: :dev},
      {:matrix, "~> 0.3.2"}
    ]
  end

  defp package do
    [
      maintainers: ["Anton Bogdanov"],
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/kortirso/elixir_learn_kit"}
    ]
  end
end
