using GLMakie

struct cockpit_visualiser
    vis_loss::Union{Observable{Vector{Float32}}, Nothing}
    vis_grad_norm::Union{Observable{Vector{Float32}}, Nothing}

    function cockpit_visualiser(vis_loss::Union{Observable{Vector{Float32}}, Nothing}, vis_grad_norm::Union{Observable{Vector{Float32}}, Nothing})
        fig = Figure()

        if !isnothing(vis_loss)
            loss_plot!(fig, vis_loss)
        end

        if !isnothing(vis_grad_norm)
            grad_norm_plot!(fig, vis_grad_norm)
        end
        
        display(fig)
        return new(vis_loss,vis_grad_norm)
    end
end

"""
    cockpit_visualiser(; vis_loss::Observable{Vector{Float32}} = nothing, vis_grad_norm::Observable{Vector{Float32}} = nothing)

Initialises the visualiser. It will take the given Observables and display a plot that updates live. 
"""
function cockpit_visualiser(; vis_loss::Union{Observable{Vector{Float32}}, Nothing} = nothing, vis_grad_norm::Union{Observable{Vector{Float32}}, Nothing} = nothing)
    return cockpit_visualiser(vis_loss, vis_grad_norm)
end

export cockpit_visualiser