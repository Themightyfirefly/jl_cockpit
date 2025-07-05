module jl_cockpit
    using GLMakie
    using LinearAlgebra: norm
    using Zygote
    using Flux
    using MLDatasets
    using Statistics: mean

    include("training_loop.jl")

    export visualiser

    export training_loop

    # Plot functions
    export loss_plot!
    export grad_norm_plot!
    export hist_1d_plot!
    export params_plot!
    export distance_plot!
    export update_size_plot!
    
    # utils
    export Datapoint
end