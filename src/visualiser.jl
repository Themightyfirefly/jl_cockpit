using GLMakie: Observable, Figure

include("plots.jl")

#TODO struct should be capitalized
struct visualiser
    datapoints::Observable{Vector{Datapoint}}
end

"""
    visualiser(; vis_loss::Observable{Vector{Float32}} = nothing, vis_grad_norm::Observable{Vector{Float32}} = nothing)

Initialises the visualiser. It will take the given Observables and display a plot that updates live. 
"""
function visualiser(; vis_loss::Bool = true, vis_grad_norm::Bool = true, vis_hist_1d = true)
    GLMakie.activate!()
    GLMakie.closeall()
    
    datapoints = Observable{Vector{Datapoint}}([])

    with_theme(theme_black()) do
        fig = Figure()
        
        vis_loss && loss_plot!(fig, datapoints)
        vis_grad_norm && grad_norm_plot!(fig, datapoints)
        vis_hist_1d && hist_1d_plot!(fig, datapoints)

        display(fig)
    end

    return visualiser(datapoints)
end

export visualiser