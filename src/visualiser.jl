using GLMakie: Observable, Figure

include("plots.jl")

#TODO struct should be capitalized
struct visualiser
    datapoints::Observable{Vector{Datapoint}}
    batch_size::Int64
end

"""
    visualiser(; vis_loss::Observable{Vector{Float32}} = nothing, vis_grad_norm::Observable{Vector{Float32}} = nothing)

Initialises the visualiser. It will take the given Observables and display a plot that updates live. 
"""
function visualiser(; batch_size = 1, vis_loss::Bool = false, vis_grad_norm::Bool = false, vis_update_size::Bool = false, vis_distance::Bool = false)
    
    datapoints = Observable{Vector{Datapoint}}([])

    fig = Figure()
    
    vis_loss && loss_plot!(fig, datapoints)
    vis_grad_norm && grad_norm_plot!(fig, datapoints)
    vis_update_size && update_size_plot!(fig, datapoints)
    vis_distance && distance_plot!(fig, datapoints)

    display(fig)

    return visualiser(datapoints, batch_size)
end

export visualiser