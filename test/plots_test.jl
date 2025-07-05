using Test
using jl_cockpit
using GLMakie: Figure, Observable

fig = Figure(size = (1920, 1080))
datapoints = Observable{Vector{Datapoint}}([])

@testset "loss plot" begin
    datapoints[] = []
    plot_data = loss_plot!(fig, datapoints)
    push!(datapoints, Datapoint(0, 0, 1, nothing, nothing))
    @test plot_data[] == []
end