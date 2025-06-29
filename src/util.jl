using GLMakie: Observable

struct Datapoint
    epoch::Int
    batch::Int
    loss::Float32
    grads #TODO define the type!
    params_before
    params_after
    params_init
end

# Extending the function push! to ensure the Observable is triggered
#
# Observables are triggered when they get assigned a value,
# but when appending the list is mutated, not assigned, and therefore the Observable must be triggered manually
import Base.push!

"""
This function updates list of Observables with specific value of FLoat32
"""
function push!(list_obs::Observable{Vector{T}}, value::T) where {T<:Real}
    push!(list_obs[], value)
    list_obs[] = list_obs[]
end

"""
This function updates Observable with datapoint
"""
function push!(obs::Observable{Vector{Datapoint}}, dp::Datapoint)
    push!(obs[], dp)
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

# If we get an array, append it to res
myflatten!(res, xs::AbstractArray) = append!(res,
xs) 

# If we get anything else (scalar values, `noting`), ignore
myflatten!(res, x) = nothing
