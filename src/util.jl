using GLMakie: Observable
using Zygote

struct Datapoint
    epoch::Int
    batch::Int
    loss::Union{Float32, Nothing}
    grads::Union{@NamedTuple{Any}, Nothing}
    params::Union{Zygote.Params{Zygote.Buffer{Any, Vector{Any}}}, Nothing}
end

# Extending the function push! to ensure the Observable is triggered
#
# Observables are triggered when they get assigned a value,
# but when appending the list is mutated, not assigned, and therefore the Observable must be triggered manually
import Base.push!

"""
    push!(list_obs::Observable{Vector{T}}, value::T) where {T<:Real}

Push number onto a Vector packaged in the given Observable and trigger Observable.
"""
function push!(list_obs::Observable{Vector{T}}, value::T) where {T<:Real}
    push!(list_obs[], value)
    list_obs[] = list_obs[]
end

"""
Push Datapoint onto a Vector packaged in the given Observable and trigger Observable.
"""
function push!(obs::Observable{Vector{Datapoint}}, dp::Datapoint)
    push!(obs[], dp)
    obs[] = obs[]
end

import Base.append!
function append!(obs::Observable{Vector{T}}, val_vector::Vector{T}) where {T<:Real}
    append!(obs[], val_vector)
    obs[] = obs[]
end

# Grad flatten as provided by Adrian

# Create empty array to write into
myflatten(x) = myflatten!(Float32[], x)
# By convention, we indicate functions what mutate inputs with an “!”.

# If `t` is a (Named)Tuple, iterate over contents and write them into `res`
function myflatten!(res, t::Union{Tuple, NamedTuple})
    for val in t
        myflatten!(res, val)
    end
    return res
end

# Expanding the function for the Parameter case
function myflatten!(res, t::Zygote.Params{Zygote.Buffer{Any, Vector{Any}}})
    for val in t
        myflatten!(res, val)
    end
    return res
end

# If we get an array, append it to res
myflatten!(res, xs::AbstractArray) = append!(res,
xs) 

# If we get anything else (scalar values, `noting`), ignore
myflatten!(res, x) = nothing
